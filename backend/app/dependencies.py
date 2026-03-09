"""
FastAPI dependency: get_current_user via Clerk JWT verification.

Clerk issues standard RS256 JWTs. We verify them against Clerk's public
JWKS endpoint (fetched once and cached) instead of a shared secret.

Flow:
  1. Extract Bearer token from Authorization header
  2. Fetch Clerk's JWKS (cached with a 1-hour TTL)
  3. Verify RS256 signature, expiry, and issuer claims
  4. Return a CurrentUser dataclass for use in route handlers

No database needed — Clerk is the source of truth for identity.
The user_id (Clerk's userId, e.g. "user_2abc123") is used as the
tenant isolation key in Qdrant metadata.
"""

import logging
import time
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()
_bearer  = HTTPBearer(auto_error=False)

# ── JWKS cache ────────────────────────────────────────────────────────────────
# Clerk's public keys rarely rotate. We cache them for 1 hour to avoid
# a network round-trip on every request while still picking up key rotations.

_jwks_cache:      dict | None = None
_jwks_fetched_at: float       = 0.0
_JWKS_TTL_SECONDS             = 3600  # 1 hour


def _get_jwks() -> dict:
    """
    Fetch Clerk's JSON Web Key Set, with a TTL-based module-level cache.

    On fetch failure:
      • If a stale cache exists, use it (availability over freshness)
      • Otherwise raise HTTP 503
    """
    global _jwks_cache, _jwks_fetched_at

    now = time.monotonic()
    if _jwks_cache and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
        return _jwks_cache

    jwks_url = f"{settings.CLERK_ISSUER}/.well-known/jwks.json"
    logger.info("Fetching Clerk JWKS from %s", jwks_url)

    try:
        resp = httpx.get(jwks_url, timeout=10)
        resp.raise_for_status()
        _jwks_cache      = resp.json()
        _jwks_fetched_at = now
        logger.info("Clerk JWKS cached (%d keys).", len(_jwks_cache.get("keys", [])))
        return _jwks_cache
    except Exception as exc:
        logger.error("Failed to fetch Clerk JWKS: %s", exc)
        if _jwks_cache:
            logger.warning("Using stale JWKS cache due to fetch failure.")
            return _jwks_cache
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Please try again shortly.",
        )


# ── Verified user dataclass ───────────────────────────────────────────────────

@dataclass
class CurrentUser:
    """
    Populated from verified Clerk JWT claims.

    user_id is Clerk's userId string (e.g. "user_2NkXyz...").
    This is stored as metadata.user_id in every Qdrant chunk,
    and used as the mandatory isolation filter on all queries.
    """
    user_id:  str        # Clerk userId — Qdrant tenant isolation key
    email:    str | None  # Primary email (None for some social logins)
    username: str | None  # Clerk username if configured


# ── Dependency ────────────────────────────────────────────────────────────────

def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer),
    ],
) -> CurrentUser:
    """
    Verify a Clerk-issued JWT and return a CurrentUser.

    Raises HTTP 401 for any of:
      • Missing Authorization header
      • Invalid / expired / tampered token
      • Issuer mismatch (token not from our Clerk instance)
      • kid not found in JWKS (after one cache-busting retry)
    """
    _unauth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ── 1. Token present? ─────────────────────────────────────────────────────
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── 2. Decode header — get key ID (kid) ───────────────────────────────────
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        logger.warning("JWT header decode failed.")
        raise _unauth

    kid = header.get("kid")
    if not kid:
        logger.warning("JWT missing 'kid' header.")
        raise _unauth

    # ── 3. Find matching public key in JWKS ───────────────────────────────────
    def _find_key(jwks: dict) -> dict | None:
        return next(
            (k for k in jwks.get("keys", []) if k.get("kid") == kid),
            None,
        )

    key = _find_key(_get_jwks())

    if key is None:
        # Key may have just rotated — bust cache and retry once
        global _jwks_fetched_at
        _jwks_fetched_at = 0.0
        key = _find_key(_get_jwks())

    if key is None:
        logger.warning("No JWKS key found for kid='%s'.", kid)
        raise _unauth

    # ── 4. Verify signature + claims ─────────────────────────────────────────
    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk tokens omit 'aud' by default
        )
    except JWTError as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise _unauth

    # ── 5. Validate issuer ────────────────────────────────────────────────────
    if payload.get("iss") != settings.CLERK_ISSUER:
        logger.warning(
            "Issuer mismatch | expected='%s' | got='%s'",
            settings.CLERK_ISSUER,
            payload.get("iss"),
        )
        raise _unauth

    # ── 6. Extract identity ───────────────────────────────────────────────────
    user_id = payload.get("sub")
    if not user_id:
        logger.warning("JWT missing 'sub' claim.")
        raise _unauth

    return CurrentUser(
        user_id=user_id,
        email=payload.get("email"),
        username=payload.get("username") or payload.get("name"),
    )


# Convenience alias so route signatures stay concise
AuthUser = Annotated[CurrentUser, Depends(get_current_user)]
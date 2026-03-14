import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.query  import router as query_router
from app.api.upload import router as upload_router
from app.config import get_settings

settings = get_settings()

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "httpx":    {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "qdrant":   {"level": "WARNING"},
        "google":   {"level": "WARNING"},
        "urllib3":  {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Document Q&A API...")
    logger.info("Auth provider: Clerk | issuer=%s", settings.CLERK_ISSUER)
    logger.info("CORS allowed origins: %s", settings.cors_origins_list)

    # Ensure upload + Qdrant dirs exist
    for directory in [Path(settings.UPLOAD_DIR), Path(settings.QDRANT_PERSIST_DIR)]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", directory.resolve())

    # Pre-fetch Clerk JWKS so the first request isn't slow
    try:
        from app.dependencies import _get_jwks
        _get_jwks()
        logger.info("Clerk JWKS pre-fetched successfully.")
    except Exception as exc:
        logger.warning("Clerk JWKS pre-fetch failed (will retry on first request): %s", exc)

    # Warm up Qdrant
    try:
        from app.services.vectorstore import get_qdrant_client, ensure_collection
        client = get_qdrant_client()
        ensure_collection()
        count  = client.count(collection_name=settings.COLLECTION_NAME, exact=True).count
        logger.info(
            "Qdrant ready — collection '%s' contains %d chunks.",
            settings.COLLECTION_NAME, count,
        )
    except Exception as exc:
        logger.warning("Qdrant warm-up failed (will retry on first request): %s", exc)

    logger.info("API startup complete.")
    yield
    logger.info("Shutting down RAG Document Q&A API.")


app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Upload PDF and DOCX documents, then ask natural-language questions. "
        "Powered by LangChain, Qdrant, and Google Gemini. "
        "Authentication via Clerk — pass the session token as Bearer."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)   # /api/documents/* — Clerk-protected
app.include_router(query_router)    # /api/query       — Clerk-protected


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    health: dict = {
        "status":   "ok",
        "version":  "2.0.0",
        "auth":     "clerk",
        "services": {"qdrant": "unknown"},
        "stats":    {"total_chunks": 0},
    }
    try:
        from app.services.vectorstore import get_qdrant_client
        client = get_qdrant_client()
        count  = client.count(collection_name=settings.COLLECTION_NAME, exact=True).count
        health["services"]["qdrant"]    = "ok"
        health["stats"]["total_chunks"] = count
    except Exception as exc:
        logger.error("Health check — Qdrant unreachable: %s", exc)
        health["status"]             = "degraded"
        health["services"]["qdrant"] = f"error: {exc}"
    return health
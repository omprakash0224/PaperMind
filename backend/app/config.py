from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    QDRANT_PERSIST_DIR: str = "./qdrant_data"
    UPLOAD_DIR: str = "./uploads"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    COLLECTION_NAME: str = "documents"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 3072 

    # ── Clerk Auth ────────────────────────────────────────────────────────────
    # Your Clerk Frontend API URL — shown in Clerk Dashboard → API Keys
    # Dev:  https://<clerk-subdomain>.clerk.accounts.dev
    # Prod: https://clerk.<yourdomain>.com
    CLERK_ISSUER: str

    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://paper-mind-rho.vercel.app",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                import json
                return json.loads(v)
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # ── Retrieval quality settings ────────────────────────────────────────────

    # Minimum cosine similarity a chunk must have to be included in context.
    # Chunks below this threshold are silently dropped before the LLM sees them.
    # Range: [0.0, 1.0]. Raise to 0.5+ for stricter, more precise answers.
    # Lower to 0.2 if you find the system saying "not enough info" too often.
    MIN_SCORE_THRESHOLD: float = 0.35

    # Number of chunks to retrieve before score filtering.
    # Retrieving more candidates (e.g. 10) then filtering gives the threshold
    # room to work without starving the LLM of context.
    RETRIEVAL_TOP_K: int = 10

    # Maximum chunks passed to the LLM after threshold filtering.
    # Keeps prompt size bounded even when many chunks pass the threshold.
    MAX_CONTEXT_CHUNKS: int = 5

    # Enable hybrid search (dense vector + BM25 keyword).
    # Requires Qdrant >= 1.7. Set False to fall back to pure semantic search.
    # Hybrid search is strongly recommended for queries with exact terms,
    # names, IDs, or acronyms that pure embeddings may miss.
    ENABLE_HYBRID_SEARCH: bool = True

    # Weight blending dense vs sparse scores (Reciprocal Rank Fusion).
    # Qdrant's RRF handles this automatically — this is the prefetch limit
    # for each sub-query (dense + sparse) before fusion.
    HYBRID_PREFETCH_LIMIT: int = 20

    # Qdrant Cloud (leave unset for local dev)
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # ── Upstash Redis (leave unset for local dev) ─────────────────────────────
    # Used to store document ingestion job status across restarts and workers.
    # Falls back to an in-memory dict when these are not set (local dev only).
    #
    # Get these from: https://console.upstash.com
    #   1. Create a Redis database
    #   2. Copy "REST URL" → UPSTASH_REDIS_REST_URL
    #   3. Copy "REST Token" → UPSTASH_REDIS_REST_TOKEN
    UPSTASH_REDIS_REST_URL:   str | None = None
    UPSTASH_REDIS_REST_TOKEN: str | None = None

    # Cloudinary (leave unset for local dev)
    CLOUDINARY_CLOUD_NAME: str | None = None
    CLOUDINARY_API_KEY: str | None = None
    CLOUDINARY_API_SECRET: str | None = None

    # @property
    # def use_cloudinary(self) -> bool:
    #     return bool(
    #         self.CLOUDINARY_CLOUD_NAME
    #         and self.CLOUDINARY_API_KEY
    #         and self.CLOUDINARY_API_SECRET
    #     )

    @property
    def use_qdrant_cloud(self) -> bool:
        return bool(self.QDRANT_URL and self.QDRANT_API_KEY)
    
    @property
    def use_redis(self) -> bool:
        """True when Upstash Redis credentials are configured."""
        return bool(self.UPSTASH_REDIS_REST_URL and self.UPSTASH_REDIS_REST_TOKEN)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
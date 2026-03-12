# Architecture — RAG Document Q&A

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Structure](#repository-structure)
3. [Backend](#backend)
   - [Entry Point & Startup](#entry-point--startup)
   - [Authentication](#authentication)
   - [API Layer](#api-layer)
   - [Ingestion Pipeline](#ingestion-pipeline)
   - [RAG Query Pipeline](#rag-query-pipeline)
   - [Vector Store](#vector-store)
   - [BM25 Sparse Encoder](#bm25-sparse-encoder)
   - [Storage Abstraction](#storage-abstraction)
   - [Status Store](#status-store)
   - [Document Parsers](#document-parsers)
   - [Configuration](#configuration)
4. [Frontend](#frontend)
   - [Auth & Middleware](#auth--middleware)
   - [Page Layout](#page-layout)
   - [Components](#components)
   - [API Client](#api-client)
5. [Data Flow](#data-flow)
   - [Upload & Ingestion Flow](#upload--ingestion-flow)
   - [Query Flow](#query-flow)
6. [Tenant Isolation](#tenant-isolation)
7. [Dev vs Production Backends](#dev-vs-production-backends)
8. [Environment Variables](#environment-variables)

---

## System Overview

```
┌────────────────────────────────────────────────────────┐
│                  Next.js Frontend                      │
│  FileUpload · DocumentList · ChatWindow · SourceCard   │
│              Clerk (Auth) · lib/api.ts                 │
└────────────────────┬───────────────────────────────────┘
                     │  HTTPS + Bearer JWT
┌────────────────────▼───────────────────────────────────┐
│                 FastAPI Backend                        │
│  /api/documents/* (upload, status, list, delete)       │
│  /api/query       (RAG question answering)             │
│  /health                                               │
│                                                        │
│  Auth: Clerk RS256 JWT ◄── JWKS cached 1 hr            │
└───┬───────────────┬───────────────────┬────────────────┘
    │               │                   │
    ▼               ▼                   ▼
 Qdrant        Upstash Redis       Cloudinary
 (vectors)     (job status)        (file storage)
    │
    ▼
 Google Gemini API
 (embeddings + LLM)
```

---

## Repository Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── query.py          # POST /api/query
│   │   │   └── upload.py         # /api/documents/* endpoints
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── bm25.py           # Pure-Python BM25 sparse encoder
│   │   │   ├── ingestion.py      # Document ingestion pipeline
│   │   │   ├── rag_chain.py      # RAG query pipeline
│   │   │   ├── status_store.py   # Redis / in-memory job status
│   │   │   ├── storage.py        # Cloudinary / local file storage
│   │   │   └── vectorstore.py    # Qdrant client & collection helpers
│   │   ├── utils/
│   │   │   └── parsers.py        # PDF (PyMuPDF) & DOCX parsers
│   │   ├── config.py             # Pydantic settings (lru_cache)
│   │   ├── dependencies.py       # Clerk JWT FastAPI dependency
│   │   └── main.py               # App factory, lifespan, CORS
│   └── .env.example
│
└── frontend/
    ├── app/
    │   ├── globals.css
    │   ├── layout.tsx            # ClerkProvider root
    │   └── page.tsx              # Main app page
    ├── components/
    │   ├── ChatWindow.tsx        # Message input & conversation
    │   ├── DocumentList.tsx      # Sidebar document browser
    │   ├── FileUpload.tsx        # Drag-and-drop uploader + poller
    │   ├── MessageBubble.tsx     # Markdown-rendered messages
    │   └── SourceCard.tsx        # Relevance-scored source chunk cards
    ├── lib/
    │   └── api.ts                # Typed fetch wrapper + all API calls
    └── middleware.ts             # Clerk Edge middleware (route protection)
```

---

## Backend

### Entry Point & Startup

**`app/main.py`**

FastAPI app is created with a `lifespan` context manager that runs on startup:

1. Creates upload and Qdrant persist directories.
2. Pre-fetches Clerk JWKS so the first request is not penalised by a network round-trip.
3. Warms up Qdrant by ensuring the collection exists and logging the chunk count.

CORS is configured from the `CORS_ORIGINS` setting (comma-separated string in env, parsed into a list).

---

### Authentication

**`app/dependencies.py`**

Clerk issues RS256 JWTs. The backend verifies them without a database, using Clerk's public JWKS endpoint.

**Verification flow:**

```
1. Extract Bearer token from Authorization header
2. Decode JWT header → get key ID (kid)
3. Find matching key in JWKS cache (TTL = 1 hour)
   └── If kid not found: bust cache, re-fetch once (handles key rotation)
4. Verify RS256 signature + expiry via python-jose
5. Validate iss == CLERK_ISSUER
6. Extract sub → user_id
```

**`AuthUser`** is a `typing.Annotated` alias used as a FastAPI dependency in every protected route:

```python
AuthUser = Annotated[CurrentUser, Depends(get_current_user)]
```

The `CurrentUser` dataclass carries `user_id`, `email`, and `username`. The `user_id` (Clerk's `sub` claim) is the tenant isolation key written into every Qdrant point.

---

### API Layer

#### `POST /api/documents/upload`

1. Validates file type (`.pdf`, `.docx`) and size (max 50 MB).
2. Saves the file to disk with a UUID-prefixed safe filename.
3. Optionally uploads to Cloudinary if credentials are set.
4. Writes an initial `queued` status record to the status store.
5. Returns HTTP **202** immediately with a `document_id`.
6. Queues a background task (`_run_ingestion`) via FastAPI's `BackgroundTasks`.

#### `GET /api/documents/status/{document_id}`

Polls the status store. Returns 404 for both missing records and records belonging to a different user (prevents user enumeration).

Status values: `queued` → `processing` → `completed` | `duplicate` | `failed`

#### `GET /api/documents`

Lists all documents for the current user by scrolling Qdrant (not the status store). Qdrant is the source of truth for completed ingestions; the status store only tracks in-flight jobs.

#### `DELETE /api/documents/{filename}`

Deletes all Qdrant chunks matching `filename + user_id`, then removes local upload files if present. Returns 404 if no chunks found.

#### `POST /api/query`

Passes the question, optional document filter, and `user_id` to the RAG pipeline. Returns the answer and source chunks.

---

### Ingestion Pipeline

**`app/services/ingestion.py`**

```
ingest_document(file_path, filename, user_id)
  │
  ├─ 0. download_for_processing()   — fetch from Cloudinary to temp file if needed
  ├─ 1. _compute_file_hash()        — SHA-256 for deduplication
  ├─ 2. _doc_hash_exists()          — Qdrant scroll filtered by (file_hash, user_id)
  │      └─ return early if duplicate
  ├─ 3. parse_document()            — PDF or DOCX → list of {text, metadata} per page
  ├─ 4. RecursiveCharacterTextSplitter
  │      chunk_size=500 tokens (tiktoken cl100k_base), overlap=50
  │      separators: ["\n\n", "\n", ". ", " "]
  │      Each chunk gets metadata: source, page, chunk_index, document_id,
  │                                file_hash, user_id
  └─ 5. _upsert_batch_with_retry()  — batches of 5 chunks
         ├─ embed_documents()       — Google Gemini (task_type=RETRIEVAL_DOCUMENT)
         ├─ passage_embed()         — BM25 TF sparse vectors
         └─ client.upsert()         — PointStruct with both dense + sparse vectors
```

**Rate limiting:** Free-tier Gemini embedding API allows 5 RPM. Batches of 5 chunks with 12-second delays between batches. Exponential backoff (15s base, doubles) on 429 errors, up to 5 retries.

**Deduplication:** Scoped per `user_id`. Two different users uploading the same file will each get their own copy (correct multi-tenant behaviour).

---

### RAG Query Pipeline

**`app/services/rag_chain.py`**

```
query(question, document_filter, user_id)
  │
  ├─ Step 1: _rewrite_query()
  │    Gemini rewrites the natural-language question into a
  │    keyword-rich search query (10–20 words).
  │    Falls back to original on any failure.
  │
  ├─ Step 2: Retrieval
  │    ├─ ENABLE_HYBRID_SEARCH=True  → _retrieve_hybrid()
  │    │    ├─ Dense prefetch:  Gemini embed_query (RETRIEVAL_QUERY task)
  │    │    ├─ Sparse prefetch: BM25 query_embed → SparseVector
  │    │    ├─ Qdrant query_points with FusionQuery(Fusion.RRF)
  │    │    └─ Normalise RRF scores to [0, 1]
  │    └─ ENABLE_HYBRID_SEARCH=False → _retrieve_dense()
  │         similarity_search_with_relevance_scores via LangChain
  │
  ├─ Step 3: _apply_score_threshold()
  │    Drop chunks below MIN_SCORE_THRESHOLD (default 0.35).
  │    Keep top MAX_CONTEXT_CHUNKS (default 5).
  │
  ├─ Step 4: _format_docs()
  │    Render chunks → "[Source: X | Page: Y]\n{text}" context string
  │
  └─ Step 5: Generation
       RAG_PROMPT | ChatGoogleGenerativeAI | StrOutputParser
       NOTE: original question (not rewritten) is sent to LLM
```

**Key design choices:**

- The rewritten query is used for **retrieval only**. The original question is passed to the LLM for generation so the answer directly addresses what the user asked.
- Score threshold runs **after** fetching `RETRIEVAL_TOP_K` candidates (default 10), giving the threshold room to filter without starving the LLM.
- RRF score normalisation ensures `MIN_SCORE_THRESHOLD` has consistent meaning in both hybrid and dense-only modes.

---

### Vector Store

**`app/services/vectorstore.py`**

**Collection schema:**

| Vector | Type | Details |
|--------|------|---------|
| `""` (unnamed) | Dense | 3072-dim, cosine distance, Google Gemini |
| `"text-sparse"` | Sparse | BM25, `Modifier.IDF` (Qdrant applies IDF server-side) |

The collection is created lazily by `ensure_collection()` on first ingestion. Toggling `ENABLE_HYBRID_SEARCH` after a collection exists requires deleting and re-ingesting.

**Singleton client:** `get_qdrant_client()` caches the `QdrantClient` at module level. Uses Qdrant Cloud when `QDRANT_URL` + `QDRANT_API_KEY` are set; otherwise local disk persistence.

**Tenant filter helper:**

```python
_build_tenant_filter(user_id, filename=None)
# Always adds: metadata.user_id == user_id
# Optionally adds: metadata.source == filename
```

Applied on every scroll, search, count, and delete operation.

---

### BM25 Sparse Encoder

**`app/services/bm25.py`**

A pure-Python, zero-dependency BM25 encoder. No fastembed, no Rust, no NLTK.

**Tokenization pipeline:** lowercase → split on non-alphanumeric → remove English stopwords → Porter-lite suffix stripping → remove tokens shorter than 2 characters.

**Token → index:** FNV-1a 32-bit hash. No vocabulary file needed; collision probability is negligible for typical document vocabularies.

**Output:** `SparseEmbedding(indices: list[int], values: list[float])` where values are raw TF counts. Qdrant applies IDF server-side via `Modifier.IDF`.

BM25 is symmetric (passage and query encoding are identical), so `passage_embed()` and `query_embed()` call the same `encode()` method.

A module-level singleton is returned by `get_bm25_encoder()`.

---

### Storage Abstraction

**`app/services/storage.py`**

| Function | Local (dev) | Cloud (prod) |
|----------|-------------|--------------|
| `upload_file()` | Returns local path unchanged | Uploads to Cloudinary, returns HTTPS URL |
| `download_for_processing()` | Returns path unchanged | Downloads URL to temp file |
| `delete_file()` | No-op (upload.py handles local cleanup) | Destroys Cloudinary asset |

`is_cloud_storage_enabled()` checks all three Cloudinary env vars. When enabled, the local file copy is deleted immediately after the Cloudinary upload succeeds.

---

### Status Store

**`app/services/status_store.py`**

Abstracts job status storage behind a simple `set_status / get_status / update_status` interface. The backend is selected at runtime based on whether Upstash credentials are configured.

| Backend | When used | Notes |
|---------|-----------|-------|
| Upstash Redis (HTTP/REST) | `UPSTASH_REDIS_REST_URL` is set | Serverless, survives restarts, multi-worker safe |
| In-memory dict | No env vars | Zero setup for local dev; does not survive restarts |

Redis keys are namespaced as `ingestion:{document_id}` with a 24-hour TTL. After expiry a 404 is returned to polling clients, which is correct (the job is long gone).

`update_status()` does a read-modify-write to merge partial updates without losing fields like `user_id` and `filename`.

---

### Document Parsers

**`app/utils/parsers.py`**

**PDF** (`parse_pdf`): PyMuPDF (`fitz`). Extracts text blocks per page using the `"blocks"` layout mode, which preserves table structures. Detects and strips repeated header/footer text (lines appearing on more than 50% of pages).

**DOCX** (`parse_docx`): `python-docx`. Extracts all paragraph text and groups into page-like sections of 30 paragraphs each. Applies the same repeated-text stripping for boilerplate headers/footers.

Both parsers return a list of `{text: str, metadata: {page_number: int, source: str}}` dicts.

---

### Configuration

**`app/config.py`**

`Settings` extends Pydantic `BaseSettings`, reading from environment variables and `.env`. An `lru_cache`-wrapped `get_settings()` function ensures a single instance is shared across the app.

Notable settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `MIN_SCORE_THRESHOLD` | `0.35` | Minimum relevance score to pass to LLM |
| `RETRIEVAL_TOP_K` | `10` | Candidates fetched before threshold filtering |
| `MAX_CONTEXT_CHUNKS` | `5` | Max chunks sent to LLM after filtering |
| `ENABLE_HYBRID_SEARCH` | `True` | Dense + BM25 via RRF vs dense-only |
| `HYBRID_PREFETCH_LIMIT` | `20` | Top-N per sub-query before RRF fusion |
| `CHUNK_SIZE` | `500` | Tokens per chunk (tiktoken cl100k_base) |
| `CHUNK_OVERLAP` | `50` | Token overlap between consecutive chunks |
| `EMBEDDING_DIM` | `3072` | Gemini `gemini-embedding-001` output dimension |

---

## Frontend

### Auth & Middleware

**`middleware.ts`** runs at the Next.js Edge before every request. All routes are protected except `/sign-in` and `/sign-up`. Unauthenticated users are redirected to `/sign-in` automatically.

**`layout.tsx`** wraps the entire app in `ClerkProvider`. No props needed — Clerk reads `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` from the environment automatically.

---

### Page Layout

**`app/page.tsx`**

Holds the three top-level state values shared across components:

- `selectedDocument` — filename string or `null` (all docs). Passed as a filter to the chat and displayed as a badge in the header.
- `refreshTrigger` — integer incremented on upload completion to tell `DocumentList` to refetch.
- `sidebarOpen` — mobile sidebar visibility toggle.

`getToken` from `useAuth()` is passed down to every child component that makes API calls, ensuring tokens are fetched fresh on every request.

---

### Components

#### `FileUpload.tsx`

Drag-and-drop zone using `react-dropzone`. Validates file type and size client-side before uploading. State machine with five phases: `idle → uploading → processing → done | error`.

After receiving the `document_id` from the upload response, it calls `pollUntilComplete()` every 2 seconds (max 120 seconds) until status is `completed` or `duplicate`. On success, it calls `onUploadComplete` to trigger a document list refresh.

#### `DocumentList.tsx`

Lists ingested documents fetched from `GET /api/documents`. Auto-refreshes every 30 seconds. Supports per-document selection (sets the `selectedDocument` filter) and deletion with a two-step confirm flow. Delete state is a discriminated union: `idle | confirming | deleting | error`.

#### `ChatWindow.tsx`

Auto-expanding textarea (max 144px), Enter to submit, Shift+Enter for newline. Appends `UserMessage`, shows a `TypingIndicator` during loading, then appends `AssistantMessage` or `ErrorMessage`. Scrolls to the bottom after each update.

#### `MessageBubble.tsx`

Renders the three message roles:
- **user** — plain text, right-aligned blue bubble.
- **assistant** — full Markdown via `react-markdown` + `remark-gfm` with custom Tailwind-styled component overrides for all block and inline elements including tables, code blocks, and task lists.
- **error** — red-bordered warning bubble.

Source chunks are rendered below the answer via `SourceList`.

#### `SourceCard.tsx`

Each retrieved chunk is shown as a card with filename, page number, a colour-coded relevance score badge (green >80%, yellow >60%, grey otherwise), a score bar, and expandable chunk text (collapsed at 160 characters).

---

### API Client

**`lib/api.ts`**

Thin typed wrapper around `fetch`. All authenticated calls receive a `GetToken` callback (`() => Promise<string | null>`) instead of storing a token in module state, preventing stale-token bugs.

`apiFetch<T>()` handles:
- Calling `getToken()` fresh on every request.
- Attaching `Authorization: Bearer {token}`.
- Parsing error bodies for a `detail` field to surface readable messages.
- Throwing `ApiError(status, detail)` for non-2xx responses.

`pollUntilComplete()` implements a recursive `setTimeout` loop with a configurable interval and deadline timeout, resolving on `completed`/`duplicate` and rejecting on `failed` or timeout.

---

## Data Flow

### Upload & Ingestion Flow

```
Browser                     Backend                      External
  │                             │                            │
  │── POST /upload ────────────►│                            │
  │   (multipart/form-data)     │── write to disk ──────────►│ (local)
  │                             │── upload_file() ──────────►│ Cloudinary
  │                             │── set_status("queued") ───►│ Redis/mem
  │◄── 202 { document_id } ─────│                            │
  │                             │                            │
  │   (background task starts)  │                            │
  │                             │── update_status("processing")
  │                             │── parse_document()         │
  │                             │── RecursiveTextSplitter    │
  │                             │── embed_documents() ──────►│ Gemini
  │                             │── passage_embed()          │ (BM25 local)
  │                             │── client.upsert() ────────►│ Qdrant
  │                             │── update_status("completed")
  │                             │                            │
  │── GET /status/{id} ────────►│                            |
  │◄── { status: "completed" } ─│                            │
```

### Query Flow

```
Browser                     Backend                      External
  │                             │                            │
  │── POST /query ─────────────►│                            │
  │   { question, filter }      │                            │
  │                             │── _rewrite_query() ───────►│ Gemini LLM
  │                             │── embed_query() ──────────►│ Gemini Embed
  │                             │── query_embed() (BM25)     │
  │                             │── query_points (RRF) ─────►│ Qdrant
  │                             │── _apply_score_threshold() │
  │                             │── RAG_PROMPT | LLM ───────►│ Gemini LLM
  │◄── { answer, sources } ─────│                            │
```

---

## Tenant Isolation

Every Qdrant point payload includes `metadata.user_id` set to the Clerk `userId`. All reads, searches, and deletes apply a mandatory `FieldCondition(key="metadata.user_id", match=user_id)` filter.

A single Qdrant collection (`documents`) serves all users. This is more scalable than per-user collections and avoids Qdrant's collection-count overhead. The filter is applied at the database level, so cross-tenant data leakage is not possible even if application logic has a bug.

The status store returns 404 for both missing records and records owned by a different user, preventing user enumeration of job IDs.

---

## Dev vs Production Backends

| Component | Local Dev | Production |
|-----------|-----------|------------|
| File storage | Local `./uploads/` directory | Cloudinary (set `CLOUDINARY_*` vars) |
| Job status | In-memory Python dict | Upstash Redis (set `UPSTASH_*` vars) |
| Vector DB | Local disk `./qdrant_data/` | Qdrant Cloud (set `QDRANT_URL` + `QDRANT_API_KEY`) |

All three switches are fully automatic — no code changes needed. Dev mode requires only a `GOOGLE_API_KEY` and a `CLERK_ISSUER` to run end-to-end.

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Gemini embeddings + LLM |
| `CLERK_ISSUER` | ✅ | Clerk Frontend API URL for JWKS verification |
| `CORS_ORIGINS` | — | Comma-separated allowed origins (default: localhost:3000) |
| `QDRANT_URL` | — | Qdrant Cloud endpoint (omit for local) |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `UPSTASH_REDIS_REST_URL` | — | Upstash REST URL (omit for in-memory) |
| `UPSTASH_REDIS_REST_TOKEN` | — | Upstash REST token |
| `CLOUDINARY_CLOUD_NAME` | — | Cloudinary cloud name (omit for local storage) |
| `CLOUDINARY_API_KEY` | — | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | — | Cloudinary API secret |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | ✅ | Clerk publishable key |
| `CLERK_SECRET_KEY` | ✅ | Clerk secret key (server-side) |
| `NEXT_PUBLIC_API_URL` | — | Backend base URL (default: `http://localhost:8000`) |
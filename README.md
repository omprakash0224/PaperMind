# 📄 PaperMind - RAG Document Q&A

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Clerk](https://img.shields.io/badge/Clerk-Auth-6C47FF?style=for-the-badge&logo=clerk&logoColor=white)
![License](https://img.shields.io/pypi/l/pytest-html-cn)

**A production-grade Retrieval-Augmented Generation (RAG) application for intelligent document Q&A.**  
Upload PDFs and DOCX files, then ask natural-language questions grounded in your content.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)

---

## Overview

PaperMind is a full-stack RAG Document Q&A application that lets you upload documents and ask questions about them in natural language. Answers are grounded exclusively in your uploaded content — no hallucinations from general knowledge.

The system uses **hybrid search** (dense semantic vectors + BM25 keyword retrieval) fused via Reciprocal Rank Fusion (RRF), **Google Gemini** for both embeddings and generation, and **Qdrant** as the vector database. Every component is designed for multi-tenant production use with per-user data isolation enforced at the database level.

The backend is **zero-setup in local dev** — only a Google API key and a Clerk issuer URL are required. Cloud services (Qdrant Cloud, Upstash Redis, Cloudinary) are optional drop-in upgrades configured via environment variables.

---

## Features

| Category | Highlights |
|---|---|
| **Ingestion** | PDF & DOCX parsing · boilerplate stripping · tiktoken chunking · SHA-256 deduplication · async HTTP 202 + status polling · exponential backoff on rate limits |
| **Retrieval** | Hybrid dense + BM25 search via Qdrant RRF · Gemini query rewriting · configurable score threshold · per-document scoping |
| **Generation** | Answers grounded exclusively in uploaded content · source attribution with filename, page, and relevance score |
| **Auth** | Clerk RS256 JWT · JWKS-cached verification · no identity database needed |
| **Multi-tenancy** | `user_id` stamped on every vector point · all DB operations filtered at the database level · enumeration-safe status API |
| **Infrastructure** | Zero-config local dev · cloud backends (Qdrant Cloud, Upstash Redis, Cloudinary) enabled via env vars · stateless workers · `/health` probe endpoint |
| **Frontend** | Drag-and-drop uploader · GFM Markdown rendering · responsive sidebar · confidence-graded source cards |

---

## Tech Stack

| | Backend | Frontend |
|---|---|---|
| **Language** | Python 3.11+ | TypeScript 5 |
| **Framework** | FastAPI | Next.js 15 (App Router) |
| **LLM / Embeddings** | Google Gemini 2.5 Flash / gemini-embedding-001 | — |
| **Vector DB** | Qdrant | — |
| **Sparse Retrieval** | Pure-Python BM25 | — |
| **Orchestration** | LangChain | — |
| **Document Parsing** | PyMuPDF · python-docx | — |
| **Auth** | Clerk (RS256 JWT / JWKS) | Clerk (`@clerk/nextjs`) |
| **Styling** | — | Tailwind CSS |
| **Markdown** | — | react-markdown + remark-gfm |
| **File Uploads** | — | react-dropzone |
| **Job Status** | Upstash Redis / in-memory | — |
| **File Storage** | Cloudinary / local disk | — |

---

## Architecture

![Architecture Diagram](docs/architecture.png)

The system is composed of three tiers:

**Frontend (Next.js)** — handles authentication via Clerk, provides the document upload UI with async polling, the document browser sidebar, and the chat interface with Markdown rendering and source attribution cards.

**Backend (FastAPI)** — exposes REST endpoints for upload, status polling, document management, and RAG queries. Ingestion runs as a background task. Authentication is stateless — Clerk JWTs are verified against a cached JWKS endpoint. All API routes enforce per-user tenant isolation.

**Data Layer** — Qdrant stores dense (Gemini) and sparse (BM25) vectors in a single collection, with `metadata.user_id` filters enforcing tenant boundaries. Upstash Redis tracks in-flight ingestion job state. Cloudinary stores raw uploaded files in production. Google Gemini provides both the embedding model and the generative LLM.

For a detailed breakdown of every service, data flow, and design decision, see [`docs/ARCHITECTURE.MD`](docs/ARCHITECTURE.MD).

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- A [Clerk](https://clerk.com/) account (free tier works)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/rag-document-qa.git
cd rag-document-qa
```

### 2. Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  

# Copy and configure environment variables
cp .env.example .env
```

Edit `backend/.env` and fill in the required values:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
CLERK_ISSUER=https://<your-clerk-subdomain>.clerk.accounts.dev

# Optional — leave unset to use local dev backends
QDRANT_URL=
QDRANT_API_KEY=
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
CORS_ORIGINS=http://localhost:3000
```

> **Finding your `CLERK_ISSUER`:** Go to Clerk Dashboard → API Keys → your Frontend API URL. It looks like `https://clerk.your-app.accounts.dev`.

Start the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 3. Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Copy and configure environment variables
cp .env.local.example .env.local
```

Edit `frontend/.env.local`:

```env
# Required — from Clerk Dashboard → API Keys
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

# Optional — defaults to http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### 4. Production deployment

For production, set the optional environment variables to enable cloud backends:

| Service | Variables | What it enables |
|---|---|---|
| Qdrant Cloud | `QDRANT_URL`, `QDRANT_API_KEY` | Persistent, scalable vector storage |
| Upstash Redis | `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | Job status survives restarts and scales across workers |
| Cloudinary | `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | Uploaded files stored in the cloud, local disk not required |

> **Note:** Toggling `ENABLE_HYBRID_SEARCH` after a Qdrant collection already exists requires deleting the collection and re-ingesting all documents, as the sparse vector index is set at collection creation time.

### Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Gemini embeddings + LLM |
| `CLERK_ISSUER` | ✅ | — | Clerk Frontend API URL for JWKS verification |
| `CORS_ORIGINS` | — | `http://localhost:3000` | Comma-separated allowed origins |
| `MIN_SCORE_THRESHOLD` | — | `0.35` | Minimum relevance score to include a chunk |
| `RETRIEVAL_TOP_K` | — | `10` | Candidate chunks fetched before score filtering |
| `MAX_CONTEXT_CHUNKS` | — | `5` | Maximum chunks sent to the LLM |
| `ENABLE_HYBRID_SEARCH` | — | `true` | Dense + BM25 RRF vs dense-only retrieval |
| `CHUNK_SIZE` | — | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | — | `50` | Token overlap between consecutive chunks |
| `QDRANT_URL` | — | — | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | — | — | Qdrant Cloud API key |
| `UPSTASH_REDIS_REST_URL` | — | — | Upstash REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | — | — | Upstash REST token |
| `CLOUDINARY_CLOUD_NAME` | — | — | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | — | — | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | — | — | Cloudinary API secret |
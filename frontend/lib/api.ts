/**
 * API client — Clerk edition.
 *
 * Key change from the manual JWT version:
 *   • No module-level _token variable or setApiToken() calls.
 *   • Instead, every call that needs auth receives the token via a
 *     callback parameter: apiFetch(path, init, getToken).
 *   • getToken is Clerk's useAuth().getToken — called fresh each time
 *     so tokens are always valid (Clerk auto-refreshes expiring tokens).
 *
 * Usage in components:
 *   const { getToken } = useAuth();
 *   const docs = await getDocuments(getToken);
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type UploadStatus = "queued" | "processing" | "completed" | "duplicate" | "failed";

export interface UploadResponse {
  document_id:  string;
  filename:     string;
  chunks_count: number;
  status:       UploadStatus;
}

export interface IngestionStatus {
  document_id:   string;
  filename:      string;
  safe_filename: string;
  status:        UploadStatus;
  chunks_count:  number;
  error?:        string;
}

export interface DocumentInfo {
  filename:     string;
  document_id:  string;
  chunks_count: number;
  uploaded_at:  string | null;
}

export interface SourceChunk {
  content: string;
  source:  string;
  page:    number;
  score:   number | null;
}

export interface QueryRequest {
  question:        string;
  document_filter: string | null;
}

export interface QueryResponse {
  answer:  string;
  sources: SourceChunk[];
}

export interface DeleteResponse {
  filename:       string;
  deleted_chunks: number;
  deleted_files:  string[];
  status:         "deleted";
}

export interface HealthResponse {
  status:   "ok" | "degraded";
  version:  string;
  services: { qdrant: string };
  stats:    { total_chunks: number };
}

// ── Clerk token getter type ───────────────────────────────────────────────────
// Matches the signature of useAuth().getToken from @clerk/nextjs
export type GetToken = () => Promise<string | null>;

// ── API error class ───────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

// ── Base client ───────────────────────────────────────────────────────────────

const BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

async function apiFetch<T>(
  path:     string,
  init:     RequestInit = {},
  getToken?: GetToken,
): Promise<T> {
  const url = `${BASE_URL}${path}`;

  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(init.headers as Record<string, string> ?? {}),
  };

  // Attach Clerk Bearer token when a getter is provided
  if (getToken) {
    const token = await getToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  }

  let response: Response;
  try {
    response = await fetch(url, { ...init, headers });
  } catch {
    throw new ApiError(
      0,
      `Cannot reach the API at ${BASE_URL}. Is the backend running?`,
    );
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string")      detail = body.detail;
      else if (typeof body?.detail === "object") detail = JSON.stringify(body.detail);
    } catch { /* body wasn't JSON */ }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

// ── Document endpoints ────────────────────────────────────────────────────────

export async function uploadDocument(file: File, getToken: GetToken): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return apiFetch<UploadResponse>("/api/documents/upload", { method: "POST", body: form }, getToken);
}

export async function getIngestionStatus(
  documentId: string,
  getToken:   GetToken,
): Promise<IngestionStatus> {
  return apiFetch<IngestionStatus>(`/api/documents/status/${documentId}`, {}, getToken);
}

export async function pollUntilComplete(
  documentId: string,
  getToken:   GetToken,
  intervalMs  = 2_000,
  timeoutMs   = 120_000,
): Promise<IngestionStatus> {
  const deadline = Date.now() + timeoutMs;

  return new Promise((resolve, reject) => {
    const tick = async () => {
      if (Date.now() > deadline) {
        reject(new ApiError(408, `Ingestion timed out after ${timeoutMs / 1000}s.`));
        return;
      }
      try {
        const s = await getIngestionStatus(documentId, getToken);
        if (s.status === "completed" || s.status === "duplicate") resolve(s);
        else if (s.status === "failed") reject(new ApiError(500, s.error ?? "Ingestion failed."));
        else setTimeout(tick, intervalMs);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

export async function getDocuments(getToken: GetToken): Promise<DocumentInfo[]> {
  return apiFetch<DocumentInfo[]>("/api/documents", {}, getToken);
}

export async function deleteDocument(filename: string, getToken: GetToken): Promise<DeleteResponse> {
  return apiFetch<DeleteResponse>(
    `/api/documents/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
    getToken,
  );
}

export async function askQuestion(
  question:        string,
  getToken:        GetToken,
  documentFilter?: string,
): Promise<QueryResponse> {
  const body: QueryRequest = { question, document_filter: documentFilter ?? null };
  return apiFetch<QueryResponse>(
    "/api/query",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    getToken,
  );
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");  // no auth needed
}
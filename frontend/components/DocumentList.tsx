"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getDocuments, deleteDocument, ApiError, type DocumentInfo, type GetToken } from "@/lib/api";

interface DocumentListProps {
  selectedDocument: string | null;
  onSelectDocument: (filename: string | null) => void;
  refreshTrigger?:  number;
  getToken:         GetToken;  // ← Clerk token getter
}

type DeleteState =
  | { phase: "idle" }
  | { phase: "confirming"; filename: string }
  | { phase: "deleting";   filename: string }
  | { phase: "error";      filename: string; message: string };

function formatChunks(n: number)  { return `${n.toLocaleString()} chunk${n !== 1 ? "s" : ""}`; }
function fileIcon(f: string)       { return f.toLowerCase().endsWith(".pdf") ? "📄" : "📝"; }
function truncateFilename(name: string, max = 28) {
  if (name.length <= max) return name;
  const ext = name.slice(name.lastIndexOf("."));
  return `${name.slice(0, max - ext.length - 1)}…${ext}`;
}

export default function DocumentList({
  selectedDocument, onSelectDocument, refreshTrigger = 0, getToken,
}: DocumentListProps) {
  const [documents,    setDocuments]    = useState<DocumentInfo[]>([]);
  const [loading,      setLoading]      = useState(true);
  const [fetchError,   setFetchError]   = useState<string | null>(null);
  const [deleteState,  setDeleteState]  = useState<DeleteState>({ phase: "idle" });

  const fetchDocuments = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    setFetchError(null);
    try {
      setDocuments(await getDocuments(getToken));
    } catch (err) {
      setFetchError(err instanceof ApiError ? err.detail : "Failed to load documents.");
    } finally {
      setLoading(false);
    }
  }, [getToken]);

  useEffect(() => { fetchDocuments(true); }, [fetchDocuments]);
  useEffect(() => { if (refreshTrigger > 0) fetchDocuments(false); }, [refreshTrigger, fetchDocuments]);
  useEffect(() => {
    const id = setInterval(() => fetchDocuments(false), 30_000);
    return () => clearInterval(id);
  }, [fetchDocuments]);

  const confirmDelete = useCallback(async (filename: string) => {
    setDeleteState({ phase: "deleting", filename });
    try {
      await deleteDocument(filename, getToken);
      onSelectDocument(null);
      setDeleteState({ phase: "idle" });
      await fetchDocuments(false);
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : "Delete failed.";
      setDeleteState({ phase: "error", filename, message });
    }
  }, [fetchDocuments, getToken, onSelectDocument]);

  return (
    <aside className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">Documents</h2>
        <button onClick={() => fetchDocuments(true)} disabled={loading} title="Refresh" className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40">
          <RefreshIcon spinning={loading} />
        </button>
      </div>

      <button
        onClick={() => onSelectDocument(null)}
        className={["flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors", selectedDocument === null ? "bg-blue-600 text-white" : "text-zinc-300 hover:bg-zinc-800"].join(" ")}
      >
        <span className="text-base">🗂️</span>
        <span className="font-medium">All documents</span>
      </button>

      <hr className="border-zinc-700" />

      <div className="flex-1 overflow-y-auto space-y-1 pr-0.5">
        {loading && documents.length === 0 ? <LoadingSkeleton /> :
         fetchError ? <ErrorBanner message={fetchError} onRetry={() => fetchDocuments(true)} /> :
         documents.length === 0 ? <EmptyState /> :
         documents.map((doc) => (
           <DocumentRow
             key={doc.document_id}
             doc={doc}
             isSelected={selectedDocument === doc.filename}
             deleteState={deleteState}
             onSelect={() => onSelectDocument(selectedDocument === doc.filename ? null : doc.filename)}
             onRequestDelete={() => setDeleteState({ phase: "confirming", filename: doc.filename })}
             onConfirmDelete={() => confirmDelete(doc.filename)}
             onCancelDelete={() => setDeleteState({ phase: "idle" })}
           />
         ))
        }
      </div>

      {documents.length > 0 && (
        <p className="text-center text-xs text-zinc-600">
          {documents.length} document{documents.length !== 1 ? "s" : ""} ·{" "}
          {documents.reduce((s, d) => s + d.chunks_count, 0).toLocaleString()} total chunks
        </p>
      )}
    </aside>
  );
}

function DocumentRow({ doc, isSelected, deleteState, onSelect, onRequestDelete, onConfirmDelete, onCancelDelete }: {
  doc: DocumentInfo; isSelected: boolean; deleteState: DeleteState;
  onSelect: () => void; onRequestDelete: () => void; onConfirmDelete: () => void; onCancelDelete: () => void;
}) {
  const isConfirming = deleteState.phase === "confirming" && deleteState.filename === doc.filename;
  const isDeleting   = deleteState.phase === "deleting"   && deleteState.filename === doc.filename;
  const hasError     = deleteState.phase === "error"      && deleteState.filename === doc.filename;

  return (
    <div className={["group rounded-lg border transition-colors", isSelected ? "border-blue-500/50 bg-blue-950/40" : "border-transparent hover:border-zinc-700 hover:bg-zinc-800/60"].join(" ")}>
      <div className="flex items-center gap-2 px-3 py-2">
        <button onClick={onSelect} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          <span className="shrink-0 text-base">{fileIcon(doc.filename)}</span>
          <div className="min-w-0">
            <p className={["truncate text-sm font-medium leading-tight", isSelected ? "text-blue-300" : "text-zinc-200"].join(" ")} title={doc.filename}>{truncateFilename(doc.filename)}</p>
            <p className="text-xs text-zinc-500">{formatChunks(doc.chunks_count)}</p>
          </div>
        </button>
        {!isDeleting && !isConfirming && (
          <button onClick={(e) => { e.stopPropagation(); onRequestDelete(); }} className={["shrink-0 rounded p-1 transition-colors text-zinc-600 hover:bg-red-950 hover:text-red-400 opacity-0 group-hover:opacity-100", hasError ? "opacity-100 text-red-400" : ""].join(" ")}>
            <TrashIcon />
          </button>
        )}
        {isDeleting && <span className="shrink-0 animate-spin text-zinc-500"><SpinnerIcon /></span>}
      </div>
      {isConfirming && (
        <div className="flex items-center justify-between gap-2 border-t border-zinc-700 px-3 py-2">
          <p className="text-xs text-zinc-400">Delete this document?</p>
          <div className="flex gap-1">
            <button onClick={onCancelDelete} className="rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-700">Cancel</button>
            <button onClick={onConfirmDelete} className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-500">Delete</button>
          </div>
        </div>
      )}
      {hasError && deleteState.phase === "error" && (
        <p className="border-t border-red-900/50 px-3 py-1.5 text-xs text-red-400">{deleteState.message}</p>
      )}
    </div>
  );
}

function EmptyState() {
  return <div className="flex flex-col items-center gap-2 py-8 text-center"><span className="text-3xl opacity-30">📭</span><p className="text-sm text-zinc-500">No documents yet.</p><p className="text-xs text-zinc-600">Upload a PDF or DOCX to get started.</p></div>;
}
function LoadingSkeleton() {
  return <div className="space-y-2 animate-pulse">{[1,2,3].map(i => <div key={i} className="flex items-center gap-2 rounded-lg px-3 py-2"><div className="h-5 w-5 rounded bg-zinc-700"/><div className="flex-1 space-y-1.5"><div className="h-3 w-3/4 rounded bg-zinc-700"/><div className="h-2.5 w-1/3 rounded bg-zinc-800"/></div></div>)}</div>;
}
function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-3 text-center"><p className="text-xs text-red-400">{message}</p><button onClick={onRetry} className="mt-2 text-xs text-red-400 underline hover:text-red-300">Retry</button></div>;
}
function TrashIcon() { return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}><path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" /></svg>; }
function RefreshIcon({ spinning }: { spinning: boolean }) { return <svg className={["h-3.5 w-3.5 transition-transform", spinning ? "animate-spin" : ""].join(" ")} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" /></svg>; }
function SpinnerIcon() { return <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>; }
"use client";

import { useCallback, useState } from "react";
import { useDropzone, FileRejection } from "react-dropzone";
import {
  uploadDocument,
  pollUntilComplete,
  ApiError,
  type GetToken,
  type IngestionStatus,
} from "@/lib/api";

type UploadPhase = "idle" | "uploading" | "processing" | "done" | "error";

interface UploadState {
  phase:    UploadPhase;
  filename: string;
  message:  string;
  chunks?:  number;
}

interface FileUploadProps {
  onUploadComplete?: (status: IngestionStatus) => void;
  getToken:          GetToken;   // ← Clerk token getter passed from page.tsx
}

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
};
const MAX_SIZE_BYTES = 50 * 1024 * 1024;

export default function FileUpload({ onUploadComplete, getToken }: FileUploadProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    phase: "idle", filename: "", message: "",
  });

  const handleFile = useCallback(
    async (file: File) => {
      setUploadState({ phase: "uploading", filename: file.name, message: "Uploading…" });

      try {
        const uploadRes = await uploadDocument(file, getToken);

        setUploadState({ phase: "processing", filename: file.name, message: "Parsing and embedding document…" });

        const finalStatus = await pollUntilComplete(uploadRes.document_id, getToken, 2_000, 120_000);

        const isDuplicate = finalStatus.status === "duplicate";
        setUploadState({
          phase:    "done",
          filename: file.name,
          message:  isDuplicate
            ? `Already indexed — ${finalStatus.chunks_count} chunks available.`
            : `Ready! Indexed ${finalStatus.chunks_count} chunks.`,
          chunks: finalStatus.chunks_count,
        });

        onUploadComplete?.(finalStatus);
        setTimeout(() => setUploadState({ phase: "idle", filename: "", message: "" }), 4_000);
      } catch (err) {
        const detail =
          err instanceof ApiError ? err.detail :
          err instanceof Error ? err.message : "Unknown error";
        setUploadState({ phase: "error", filename: file.name, message: detail });
      }
    },
    [getToken, onUploadComplete],
  );

  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]) => {
      if (rejections.length > 0) {
        const reason = rejections[0].errors[0];
        const detail =
          reason.code === "file-too-large"    ? "File exceeds 50 MB limit." :
          reason.code === "file-invalid-type" ? "Only .pdf and .docx files are accepted." :
          reason.message;
        setUploadState({ phase: "error", filename: rejections[0].file.name, message: detail });
        return;
      }
      if (accepted.length > 0) handleFile(accepted[0]);
    },
    [handleFile],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
    disabled: uploadState.phase === "uploading" || uploadState.phase === "processing",
  });

  const { phase, filename, message } = uploadState;
  const isActive = phase === "uploading" || phase === "processing";
  const borderColor =
    isDragReject   ? "border-red-400" :
    isDragActive   ? "border-blue-400" :
    phase === "done"  ? "border-green-400" :
    phase === "error" ? "border-red-400" :
    "border-zinc-600 hover:border-zinc-400";

  return (
    <div className="w-full space-y-3">
      <div
        {...getRootProps()}
        className={[
          "relative flex flex-col items-center justify-center gap-2",
          "rounded-xl border-2 border-dashed px-6 py-8 text-center",
          "transition-colors duration-150 outline-none",
          isActive ? "cursor-wait opacity-70" : "cursor-pointer",
          borderColor,
        ].join(" ")}
      >
        <input {...getInputProps()} />
        <DropIcon phase={phase} isDragActive={isDragActive} />
        <p className="text-sm font-medium text-zinc-200">
          {isDragActive && !isDragReject ? "Drop it here…" :
           isDragReject                  ? "File type not supported" :
           isActive && phase === "uploading" ? "Uploading…" :
           isActive                      ? "Processing…" :
           "Drop a PDF or DOCX here"}
        </p>
        {!isActive && phase === "idle" && (
          <p className="text-xs text-zinc-500">
            or <span className="text-blue-400 underline underline-offset-2">click to browse</span> · max 50 MB
          </p>
        )}
        {isActive && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-zinc-900/60">
            <Spinner />
          </div>
        )}
      </div>

      {phase !== "idle" && <Toast phase={phase} filename={filename} message={message} />}
    </div>
  );
}

function DropIcon({ phase, isDragActive }: { phase: UploadPhase; isDragActive: boolean }) {
  if (phase === "done")
    return <svg className="h-8 w-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
  if (phase === "error")
    return <svg className="h-8 w-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>;
  return (
    <svg className={["h-8 w-8 transition-transform duration-150", isDragActive ? "scale-110 text-blue-400" : "text-zinc-400"].join(" ")} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m.75 12l3 3m0 0l3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  );
}

function Toast({ phase, filename, message }: { phase: UploadPhase; filename: string; message: string }) {
  const styles: Record<UploadPhase, string> = {
    idle:       "bg-zinc-800 border-zinc-700 text-zinc-300",
    uploading:  "bg-blue-950 border-blue-700 text-blue-200",
    processing: "bg-blue-950 border-blue-700 text-blue-200",
    done:       "bg-green-950 border-green-700 text-green-200",
    error:      "bg-red-950 border-red-700 text-red-200",
  };
  const icons: Record<UploadPhase, string> = { idle: "", uploading: "⏫", processing: "⚙️", done: "✅", error: "❌" };
  return (
    <div role="status" aria-live="polite" className={["flex items-start gap-3 rounded-lg border px-4 py-3 text-sm transition-all duration-300", styles[phase]].join(" ")}>
      <span className="mt-px shrink-0 text-base leading-none">
        {phase === "uploading" || phase === "processing" ? <Spinner size="sm" /> : icons[phase]}
      </span>
      <div className="min-w-0">
        <p className="truncate font-medium">{filename}</p>
        <p className="mt-0.5 text-xs opacity-80">{message}</p>
      </div>
    </div>
  );
}

function Spinner({ size = "md" }: { size?: "sm" | "md" }) {
  const dim = size === "sm" ? "h-4 w-4" : "h-6 w-6";
  return (
    <svg className={`${dim} animate-spin text-current`} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
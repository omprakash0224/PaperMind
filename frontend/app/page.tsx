"use client";

import { useCallback, useState } from "react";
import { useAuth, useUser, UserButton } from "@clerk/nextjs";
import DocumentList from "@/components/DocumentList";
import FileUpload from "@/components/FileUpload";
import ChatWindow from "@/components/ChatWindow";
import type { IngestionStatus } from "@/lib/api";

/**
 * Main app page.
 *
 * Auth is handled entirely by Clerk middleware (middleware.ts) —
 * unauthenticated users never reach this component.
 *
 * useAuth().getToken is passed down to every API call so Clerk can
 * transparently refresh the session token when it's about to expire.
 *
 * UserButton renders Clerk's pre-built avatar + dropdown (profile,
 * sign-out, manage account) — zero custom UI needed.
 */
export default function Home() {
  const { getToken }  = useAuth();
  const { user }      = useUser();

  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [refreshTrigger,   setRefreshTrigger]   = useState(0);
  const [sidebarOpen,      setSidebarOpen]       = useState(false);

  const handleUploadComplete = useCallback((_status: IngestionStatus) => {
    setRefreshTrigger((n) => n + 1);
  }, []);

  const handleSelectDocument = useCallback((filename: string | null) => {
    setSelectedDocument(filename);
    setSidebarOpen(false);
  }, []);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-zinc-950 text-zinc-100">

      {/* ── Title bar ───────────────────────────────────────────────────────── */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-zinc-800 bg-zinc-900 px-4">
        <div className="flex items-center gap-2.5">
          {/* Mobile hamburger */}
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="rounded p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 lg:hidden"
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            {sidebarOpen ? <XIcon /> : <MenuIcon />}
          </button>

          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-blue-500 to-violet-600 text-xs font-bold select-none">
            Q
          </div>
          <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
            Document Q&amp;A
          </h1>
        </div>

        <div className="flex items-center gap-3">
          {/* Document filter badge */}
          {selectedDocument && (
            <div className="flex items-center gap-1.5 rounded-full border border-blue-800/60 bg-blue-950/50 px-3 py-1">
              <span className="text-[10px] text-blue-400">Filtering:</span>
              <span
                className="max-w-[160px] truncate text-[11px] font-medium text-blue-300"
                title={selectedDocument}
              >
                {selectedDocument}
              </span>
              <button
                onClick={() => setSelectedDocument(null)}
                className="ml-0.5 text-blue-500 hover:text-blue-300 transition-colors"
                aria-label="Clear document filter"
              >
                <XSmallIcon />
              </button>
            </div>
          )}

          {/* Username — hidden on mobile, shown on sm+ */}
          {user && (
            <span className="hidden text-xs text-zinc-500 sm:block">
              {user.username ?? user.firstName ?? user.emailAddresses[0]?.emailAddress}
            </span>
          )}

          {/*
            UserButton is Clerk's pre-built avatar component.
            Clicking it opens a dropdown with:
              • Manage account (profile, email, password)
              • Sign out
            Appearance can be customised — see Clerk docs.
          */}
          <UserButton
            appearance={{
              elements: {
                avatarBox: "h-7 w-7",
              },
            }}
          />
        </div>
      </header>

      {/* ── Body ────────────────────────────────────────────────────────────── */}
      <div className="relative flex flex-1 overflow-hidden">

        {sidebarOpen && (
          <div
            className="absolute inset-0 z-20 bg-black/60 lg:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-hidden
          />
        )}

        <aside
          className={[
            "flex w-72 shrink-0 flex-col gap-4 overflow-hidden",
            "border-r border-zinc-800 bg-zinc-900",
            "lg:relative lg:flex lg:translate-x-0",
            "absolute inset-y-0 left-0 z-30 transition-transform duration-200",
            sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          ].join(" ")}
        >
          <div className="shrink-0 border-b border-zinc-800 p-4">
            <p className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-zinc-500">
              Upload Document
            </p>
            {/* Pass getToken so FileUpload can authenticate uploads */}
            <FileUpload onUploadComplete={handleUploadComplete} getToken={getToken} />
          </div>

          <div className="min-h-0 flex-1 overflow-hidden px-4 pb-4">
            <DocumentList
              selectedDocument={selectedDocument}
              onSelectDocument={handleSelectDocument}
              refreshTrigger={refreshTrigger}
              getToken={getToken}
            />
          </div>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <ChatWindow selectedDocument={selectedDocument} getToken={getToken} />
        </main>
      </div>
    </div>
  );
}

function MenuIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    </svg>
  );
}
function XIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}
function XSmallIcon() {
  return (
    <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

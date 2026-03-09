"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { SourceChunk } from "@/lib/api";
import { SourceList } from "@/components/SourceCard";    // add this import

// ── Types ─────────────────────────────────────────────────────────────────────

export interface UserMessage {
  role: "user";
  id: string;
  text: string;
}

export interface AssistantMessage {
  role: "assistant";
  id: string;
  text: string;
  sources: SourceChunk[];
}

export interface ErrorMessage {
  role: "error";
  id: string;
  text: string;
}

export type Message = UserMessage | AssistantMessage | ErrorMessage;

// ── Markdown component overrides ──────────────────────────────────────────────
// Each key maps a HTML tag to a Tailwind-styled React element.
// remark-gfm adds: tables, strikethrough, task lists, autolinks.

const markdownComponents: Components = {
  // ── Block elements ──────────────────────────────────────────────────────────
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 leading-relaxed text-zinc-100">{children}</p>
  ),
  h1: ({ children }) => (
    <h1 className="mb-2 mt-3 text-base font-bold text-white first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-3 text-sm font-bold text-white first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-1.5 mt-2 text-sm font-semibold text-zinc-200 first:mt-0">
      {children}
    </h3>
  ),

  // ── Lists ───────────────────────────────────────────────────────────────────
  ul: ({ children }) => (
    <ul className="mb-2 ml-4 space-y-0.5 list-disc marker:text-zinc-500">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 ml-4 space-y-0.5 list-decimal marker:text-zinc-500">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="pl-1 text-zinc-200 leading-relaxed">{children}</li>
  ),

  // ── Task list checkbox (remark-gfm) ─────────────────────────────────────────
  input: ({ type, checked }) =>
    type === "checkbox" ? (
      <input
        type="checkbox"
        checked={checked}
        readOnly
        className="mr-1.5 accent-blue-500"
      />
    ) : null,

  // ── Inline elements ─────────────────────────────────────────────────────────
  strong: ({ children }) => (
    <strong className="font-semibold text-white">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-zinc-300">{children}</em>
  ),
  del: ({ children }) => (
    <del className="line-through text-zinc-500">{children}</del>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-400 underline underline-offset-2 hover:text-blue-300 transition-colors"
    >
      {children}
    </a>
  ),

  // ── Code ────────────────────────────────────────────────────────────────────
  code: ({ className, children, ...props }) => {
    const isBlock = Boolean(className);        // fenced ``` blocks have a language class
    if (isBlock) {
      return (
        <code
          className={[
            "block w-full overflow-x-auto rounded-lg bg-zinc-950 px-4 py-3",
            "font-mono text-xs leading-relaxed text-emerald-300",
            className,
          ].join(" ")}
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className="rounded bg-zinc-700/70 px-1.5 py-0.5 font-mono text-xs text-emerald-300"
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="mb-3 mt-1 overflow-hidden rounded-lg">{children}</pre>
  ),

  // ── Blockquote ───────────────────────────────────────────────────────────────
  blockquote: ({ children }) => (
    <blockquote className="mb-2 border-l-2 border-blue-500 pl-3 text-zinc-400 italic">
      {children}
    </blockquote>
  ),

  // ── Horizontal rule ──────────────────────────────────────────────────────────
  hr: () => <hr className="my-3 border-zinc-700" />,

  // ── Table (remark-gfm) ───────────────────────────────────────────────────────
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto rounded-lg border border-zinc-700">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-zinc-800 text-zinc-300">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-zinc-700/60">{children}</tbody>
  ),
  tr: ({ children }) => (
    <tr className="transition-colors hover:bg-zinc-800/40">{children}</tr>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 text-left font-semibold text-zinc-200 whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-2 text-zinc-300">{children}</td>
  ),
};

// ── AssistantMarkdown ─────────────────────────────────────────────────────────

function AssistantMarkdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={markdownComponents}
    >
      {text}
    </ReactMarkdown>
  );
}

// ── MessageBubble (default export) ────────────────────────────────────────────

export default function MessageBubble({ message }: { message: Message }) {
  // ── User ───────────────────────────────────────────────────────────────────
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm">
          {/* User messages are plain text — no markdown rendering needed */}
          <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
        </div>
      </div>
    );
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (message.role === "error") {
    return (
      <div className="flex justify-start">
        <div className="flex max-w-[80%] items-start gap-2.5 rounded-2xl rounded-tl-sm border border-red-900/60 bg-red-950/50 px-4 py-3">
          <span className="mt-0.5 shrink-0 text-base" aria-hidden>⚠️</span>
          <p className="text-sm leading-relaxed text-red-300">{message.text}</p>
        </div>
      </div>
    );
  }

  // ── Assistant ──────────────────────────────────────────────────────────────
  return (
    <div className="flex justify-start" role="article" aria-label="Assistant message">
      <div className="max-w-[85%] space-y-3">
        {/* Avatar + markdown bubble */}
        <div className="flex items-start gap-2.5">
          {/* Gradient avatar */}
          <div
            aria-hidden
            className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-violet-600 text-xs select-none"
          >
            ✦
          </div>

          {/* Answer bubble — react-markdown renders inside here */}
          <div className="rounded-2xl rounded-tl-sm bg-zinc-800 px-4 py-3 shadow-sm">
            <AssistantMarkdown text={message.text} />
          </div>
        </div>

        {/* Source cards */}
        {message.sources.length > 0 && (
          <SourceList sources={message.sources} /> 
        )}
      </div>
    </div>
  );
}
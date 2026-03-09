"use client";

import { useState } from "react";
import type { SourceChunk } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface SourceCardProps {
  source: SourceChunk;
  index: number;
}

interface SourceListProps {
  sources: SourceChunk[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const PREVIEW_LENGTH = 160;

function scoreColor(score: number): {
  badge: string;
  bar: string;
  label: string;
} {
  if (score > 0.8)
    return {
      badge: "bg-green-950 text-green-400 ring-green-900/40",
      bar:   "bg-green-500",
      label: "High relevance",
    };
  if (score > 0.6)
    return {
      badge: "bg-yellow-950 text-yellow-400 ring-yellow-900/40",
      bar:   "bg-yellow-500",
      label: "Medium relevance",
    };
  return {
    badge: "bg-zinc-800 text-zinc-500 ring-zinc-700/40",
    bar:   "bg-zinc-500",
    label: "Low relevance",
  };
}

function fileExtIcon(filename: string): string {
  return filename.toLowerCase().endsWith(".pdf") ? "📄" : "📝";
}

function truncateFilename(name: string, max = 32): string {
  if (name.length <= max) return name;
  const ext  = name.slice(name.lastIndexOf("."));
  const stem = name.slice(0, max - ext.length - 1);
  return `${stem}…${ext}`;
}

// ── Score bar ─────────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number }) {
  const { bar, label } = scoreColor(score);
  const pct = Math.round(score * 100);

  return (
    <div className="flex items-center gap-2" title={`${label}: ${pct}%`}>
      {/* Track */}
      <div className="h-1 w-16 overflow-hidden rounded-full bg-zinc-700">
        {/* Fill */}
        <div
          className={`h-full rounded-full transition-all duration-500 ${bar}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="tabular-nums text-[10px] text-zinc-500">{pct}%</span>
    </div>
  );
}

// ── SourceCard ────────────────────────────────────────────────────────────────

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const hasMore   = source.content.length > PREVIEW_LENGTH;
  const displayed = expanded
    ? source.content
    : source.content.slice(0, PREVIEW_LENGTH);

  const colors = source.score !== null ? scoreColor(source.score) : null;

  return (
    <article
      aria-label={`Source ${index}: ${source.source}`}
      className={[
        "group rounded-xl border bg-zinc-900/80 text-xs transition-colors duration-150",
        "hover:border-zinc-600",
        colors ? `ring-1 ${colors.badge.split(" ").find(c => c.startsWith("ring-"))}` : "",
        "border-zinc-700/60",
      ].join(" ")}
    >
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-3 px-3 pt-2.5 pb-2">
        {/* Left: index + filename + page */}
        <div className="flex min-w-0 items-center gap-2">
          {/* Index badge */}
          <span
            aria-hidden
            className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-700 text-[10px] font-bold text-zinc-300"
          >
            {index}
          </span>

          {/* File icon */}
          <span aria-hidden className="shrink-0 text-sm leading-none">
            {fileExtIcon(source.source)}
          </span>

          {/* Filename */}
          <span
            className="truncate font-medium text-zinc-200"
            title={source.source}
          >
            {truncateFilename(source.source)}
          </span>

          {/* Page pill */}
          {source.page > 0 && (
            <span
              className="shrink-0 rounded-md bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-500"
              title={`Page ${source.page}`}
            >
              p.{source.page}
            </span>
          )}
        </div>

        {/* Right: score badge */}
        {source.score !== null && colors && (
          <span
            title={`${colors.label}: ${Math.round(source.score * 100)}%`}
            className={[
              "shrink-0 rounded-md px-1.5 py-0.5 font-mono font-semibold",
              "ring-1 tabular-nums",
              colors.badge,
            ].join(" ")}
          >
            {Math.round(source.score * 100)}%
          </span>
        )}
      </div>

      {/* Score bar — visible only when score is present */}
      {source.score !== null && (
        <div className="px-3 pb-2">
          <ScoreBar score={source.score} />
        </div>
      )}

      {/* ── Divider ─────────────────────────────────────────────────────────── */}
      <div className="border-t border-zinc-700/50" />

      {/* ── Chunk text ──────────────────────────────────────────────────────── */}
      <div className="px-3 py-2.5">
        <p className="leading-relaxed text-zinc-400">
          {displayed}
          {hasMore && !expanded && (
            <span className="text-zinc-600">…</span>
          )}
        </p>

        {/* Expand / collapse */}
        {hasMore && (
          <button
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            className={[
              "mt-1.5 flex items-center gap-1 text-blue-500",
              "transition-colors hover:text-blue-400",
              "focus:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 rounded",
            ].join(" ")}
          >
            <ChevronIcon expanded={expanded} />
            {expanded ? "Show less" : "Show more"}
          </button>
        )}
      </div>
    </article>
  );
}

// ── SourceList (convenience wrapper used by MessageBubble) ────────────────────

export function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) return null;

  return (
    <div className="ml-8 space-y-2">
      {/* Section header */}
      <div className="flex items-center gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">
          Sources
        </p>
        <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">
          {sources.length}
        </span>
      </div>

      {/* Cards */}
      <div className="space-y-1.5">
        {sources.map((src, i) => (
          <SourceCard
            key={`${src.source}-${src.page}-${i}`}
            source={src}
            index={i + 1}
          />
        ))}
      </div>
    </div>
  );
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={[
        "h-3 w-3 transition-transform duration-150",
        expanded ? "rotate-180" : "",
      ].join(" ")}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2.5}
      aria-hidden
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

// ── Default export ────────────────────────────────────────────────────────────

export default SourceCard;
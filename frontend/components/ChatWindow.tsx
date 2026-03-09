"use client";

import { useCallback, useEffect, useRef, useState, KeyboardEvent } from "react";
import { askQuestion, ApiError, type GetToken } from "@/lib/api";
import MessageBubble, { type Message, type UserMessage, type AssistantMessage, type ErrorMessage } from "@/components/MessageBubble";

interface ChatWindowProps {
  selectedDocument: string | null;
  getToken:         GetToken;   // ← Clerk token getter
}

function uid() { return Math.random().toString(36).slice(2, 10); }

export default function ChatWindow({ selectedDocument, getToken }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = useCallback(async () => {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg: UserMessage = { role: "user", id: uid(), text: question };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    if (inputRef.current) inputRef.current.style.height = "auto";

    try {
      const result = await askQuestion(question, getToken, selectedDocument ?? undefined);
      const assistantMsg: AssistantMessage = {
        role: "assistant", id: uid(), text: result.answer, sources: result.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const detail =
        err instanceof ApiError ? err.detail :
        err instanceof Error    ? err.message : "Unexpected error. Please try again.";
      const errorMsg: ErrorMessage = { role: "error", id: uid(), text: detail };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [input, loading, selectedDocument, getToken]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
    },
    [handleSubmit],
  );

  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }, []);

  return (
    <div className="flex h-full flex-col">
      {selectedDocument && (
        <div className="flex items-center gap-2 border-b border-zinc-700/60 bg-blue-950/30 px-4 py-2 text-xs text-blue-300">
          <span aria-hidden>🔍</span>
          <span>Searching only in <span className="font-medium">{selectedDocument}</span></span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && !loading && <EmptyState hasFilter={!!selectedDocument} />}
        {messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} aria-hidden />
      </div>

      <div className="border-t border-zinc-700/60 bg-zinc-900 px-4 py-3">
        <div className={["flex items-end gap-3 rounded-xl border px-4 py-3 transition-colors", loading ? "border-zinc-700 bg-zinc-800/50" : "border-zinc-700 bg-zinc-800 focus-within:border-blue-500"].join(" ")}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={selectedDocument ? `Ask about ${selectedDocument}…` : "Ask a question about your documents…"}
            disabled={loading}
            rows={1}
            aria-label="Question input"
            className="flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none disabled:opacity-50 max-h-36 overflow-y-auto leading-relaxed"
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || loading}
            title="Send (Enter)"
            className={["shrink-0 rounded-lg p-2 transition-all", input.trim() && !loading ? "bg-blue-600 text-white hover:bg-blue-500 active:scale-95" : "bg-zinc-700 text-zinc-500 cursor-not-allowed"].join(" ")}
          >
            {loading ? <MiniSpinner /> : <SendIcon />}
          </button>
        </div>
        <p className="mt-1.5 text-center text-xs text-zinc-600">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}

function EmptyState({ hasFilter }: { hasFilter: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-800 text-3xl shadow-inner">💬</div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-zinc-300">{hasFilter ? "Ask about this document" : "Ask anything"}</p>
        <p className="text-xs text-zinc-500 max-w-xs">{hasFilter ? "Your question will be answered using only the selected document." : "Upload a document, then ask questions. Answers are grounded in your files."}</p>
      </div>
      <div className="flex flex-wrap justify-center gap-2 max-w-sm">
        {["What is the main topic?", "Summarise key findings", "What are the risks mentioned?"].map((hint) => (
          <span key={hint} className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-500">{hint}</span>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start" aria-label="Assistant is thinking…" role="status">
      <div className="flex items-start gap-2.5">
        <div aria-hidden className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-violet-600 text-xs">✦</div>
        <div className="rounded-2xl rounded-tl-sm bg-zinc-800 px-4 py-3.5">
          <div className="flex items-center gap-1.5" aria-hidden>
            {[0, 1, 2].map((i) => <span key={i} className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: `${i * 150}ms` }} />)}
          </div>
        </div>
      </div>
    </div>
  );
}

function SendIcon() {
  return <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden><path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" /></svg>;
}
function MiniSpinner() {
  return <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>;
}
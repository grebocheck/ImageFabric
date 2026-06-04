import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Model, RagDocument, RagSearchResponse, RagStatus } from "../types";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-emerald-500";
const RAG_SYSTEM = "You are a careful local RAG assistant. Answer from the retrieved context when possible. Cite bracketed source numbers like [1]. If the context is insufficient, say what is missing.";

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

function excerpt(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, 180);
}

function buildPrompt(query: string, context: string): string {
  return `Retrieved context:\n\n${context || "(no retrieved context)"}\n\nQuestion:\n\n${query.trim()}`;
}

export function RagPanel({
  models,
  onOpenChat,
}: {
  models: Model[];
  onOpenChat: (conversationId: string, jobId: string) => void;
}) {
  const llmModels = useMemo(() => models.filter((m) => m.job_type === "llm"), [models]);
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [docs, setDocs] = useState<RagDocument[]>([]);
  const [docQuery, setDocQuery] = useState("");
  const [embedModelId, setEmbedModelId] = useState("");
  const [llmModelId, setLlmModelId] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [upload, setUpload] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [search, setSearch] = useState<RagSearchResponse | null>(null);
  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    api.ragStatus().then((s) => {
      setStatus(s);
      setEmbedModelId((prev) => prev || s.models[0]?.id || "");
    }).catch(() => {});
    refreshDocs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!llmModelId && llmModels[0]) setLlmModelId(llmModels[0].id);
  }, [llmModelId, llmModels]);

  useEffect(() => {
    const h = window.setTimeout(() => refreshDocs(docQuery), 180);
    return () => window.clearTimeout(h);
  }, [docQuery]);

  function refreshDocs(q = docQuery) {
    api.listRagDocuments(q).then(setDocs).catch(() => setDocs([]));
  }

  const ready = Boolean(status?.ready && embedModelId);

  async function indexText() {
    if (!ready || !content.trim() || busy) return;
    setBusy("index");
    setNote("");
    try {
      const doc = await api.createRagDocument({
        title: title || "Pasted document",
        content,
        source: "paste",
        model_id: embedModelId,
      });
      setDocs((prev) => [doc, ...prev]);
      setTitle("");
      setContent("");
      setNote(`indexed ${doc.chunks_count} chunks`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "index failed");
    } finally {
      setBusy("");
    }
  }

  async function uploadFile() {
    if (!ready || !upload || busy) return;
    setBusy("upload");
    setNote("");
    try {
      const doc = await api.uploadRagDocument({ file: upload, model_id: embedModelId });
      setDocs((prev) => [doc, ...prev]);
      setUpload(null);
      setNote(`indexed ${doc.chunks_count} chunks`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "upload failed");
    } finally {
      setBusy("");
    }
  }

  async function remove(id: string) {
    await api.deleteRagDocument(id).catch((err) => setNote(err instanceof Error ? err.message : "delete failed"));
    setDocs((prev) => prev.filter((doc) => doc.id !== id));
  }

  async function runSearch() {
    if (!ready || !query.trim() || busy) return;
    setBusy("search");
    setNote("");
    try {
      const next = await api.searchRag({ query, top_k: topK, model_id: embedModelId });
      setSearch(next);
      setNote(`${next.results.length} matches`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "search failed");
    } finally {
      setBusy("");
    }
  }

  async function askLlm() {
    if (!llmModelId || !query.trim() || busy) return;
    const context = search?.context || "";
    setBusy("llm");
    setNote("");
    try {
      const conv = await api.createConversation({
        title: `RAG: ${query.trim().slice(0, 80)}`,
        model_id: llmModelId,
        system: RAG_SYSTEM,
        params: { temperature: 0.2, max_tokens: 1400 },
      });
      const res = await api.sendChatMessage(conv.id, {
        content: buildPrompt(query, context),
        model_id: llmModelId,
        system: RAG_SYSTEM,
        temperature: 0.2,
        max_tokens: 1400,
      });
      onOpenChat(res.conversation.id, res.job_id);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "LLM request failed");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="grid h-full grid-cols-[320px_1fr_360px] gap-3">
      <aside className="flex min-h-0 flex-col rounded-lg border border-white/10">
        <div className="border-b border-white/10 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-white/75">Documents</div>
              <div className="text-xs text-white/35">{ready ? "embed ready" : "waiting for embed model"}</div>
            </div>
            <span className="rounded bg-white/10 px-1.5 py-0.5 text-xs text-white/45">{docs.length}</span>
          </div>
          <input
            value={docQuery}
            onChange={(e) => setDocQuery(e.target.value)}
            placeholder="search documents"
            className={`${field} text-xs`}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {docs.length === 0 && <div className="px-1 py-2 text-xs text-white/30">no documents</div>}
          {docs.map((doc) => (
            <div key={doc.id} className="mb-1 rounded-md px-2 py-2 hover:bg-white/5">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-white/80" title={doc.title}>{doc.title}</div>
                  <div className="mt-0.5 truncate text-xs text-white/35" title={doc.source ?? ""}>
                    {doc.source || "local"} · {doc.chunks_count} chunks
                  </div>
                </div>
                <button
                  onClick={() => void remove(doc.id)}
                  className="shrink-0 rounded px-1.5 py-0.5 text-xs text-red-300/70 hover:bg-red-400/10 hover:text-red-200"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <section className="flex min-h-0 flex-col rounded-lg border border-white/10">
        <div className="grid grid-cols-[1fr_auto] gap-3 border-b border-white/10 p-3">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void runSearch(); }}
            placeholder="Ask against indexed documents"
            className={field}
          />
          <button
            onClick={() => void runSearch()}
            disabled={!ready || !query.trim() || Boolean(busy)}
            className="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30"
          >
            {busy === "search" ? "Searching..." : "Search"}
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {search?.results.length ? search.results.map((item, idx) => (
            <div key={item.chunk_id} className="mb-2 rounded-md border border-white/10 bg-black/15 p-3">
              <div className="mb-1 flex items-center justify-between gap-2 text-xs">
                <span className="min-w-0 truncate font-medium text-white/60" title={item.document_title}>
                  [{idx + 1}] {item.document_title}
                </span>
                <span className="shrink-0 font-mono text-white/35">{item.score.toFixed(3)}</span>
              </div>
              <div className="text-sm leading-6 text-white/70">{excerpt(item.text)}</div>
            </div>
          )) : (
            <div className="text-sm text-white/30">{search ? "no matches" : ""}</div>
          )}
        </div>
      </section>

      <aside className="flex min-h-0 flex-col gap-3 rounded-lg border border-white/10 p-4">
        <div className="space-y-1.5 rounded-md border border-white/10 bg-black/20 p-3 text-xs">
          <Row label="Binary" value={status?.binary_exists ? "found" : "missing"} />
          <Row label="Models" value={status?.models_dir ?? "..."} mono />
          <Row label="Port" value={status ? String(status.port) : "..."} />
        </div>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Embedding</div>
          <select value={embedModelId} onChange={(e) => setEmbedModelId(e.target.value)} className={`${field} mt-1`}>
            {(status?.models ?? []).length === 0 && <option value="">no embed models</option>}
            {(status?.models ?? []).map((m) => <option key={m.id} value={m.id}>{m.name} ({size(m.size_bytes)})</option>)}
          </select>
        </label>

        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">LLM</div>
          <select value={llmModelId} onChange={(e) => setLlmModelId(e.target.value)} className={`${field} mt-1`}>
            {llmModels.length === 0 && <option value="">no LLM models</option>}
            {llmModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>

        <div className="grid grid-cols-2 gap-2">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Top K</div>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
              className={`${field} mt-1`}
            />
          </label>
          <div className="flex items-end">
            <button
              onClick={() => void askLlm()}
              disabled={!llmModelId || !query.trim() || Boolean(busy)}
              className="w-full rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30"
            >
              {busy === "llm" ? "Sending..." : "Send to LLM"}
            </button>
          </div>
        </div>

        <div className="border-t border-white/10 pt-3">
          <div className="mb-2 text-xs uppercase tracking-wide text-white/40">Index text</div>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="title"
            className={`${field} mb-2`}
          />
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="paste text"
            className={`${field} h-28 resize-none`}
          />
          <button
            onClick={() => void indexText()}
            disabled={!ready || !content.trim() || Boolean(busy)}
            className="mt-2 w-full rounded-md border border-emerald-400/30 px-3 py-1.5 text-sm text-emerald-200 hover:bg-emerald-500/10 disabled:opacity-30"
          >
            {busy === "index" ? "Indexing..." : "Index"}
          </button>
        </div>

        <div className="border-t border-white/10 pt-3">
          <input
            type="file"
            accept=".txt,.md,.json,.csv"
            onChange={(e) => setUpload(e.target.files?.[0] ?? null)}
            className={`${field} file:mr-3 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-white/70`}
          />
          <button
            onClick={() => void uploadFile()}
            disabled={!ready || !upload || Boolean(busy)}
            className="mt-2 w-full rounded-md border border-white/15 px-3 py-1.5 text-sm text-white/70 hover:bg-white/10 disabled:opacity-30"
          >
            {busy === "upload" ? "Uploading..." : "Upload"}
          </button>
        </div>

        <span className="min-h-4 truncate text-xs text-white/35" title={note}>{note}</span>
      </aside>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[62px_1fr] gap-2">
      <span className="text-white/35">{label}</span>
      <span className={`truncate text-white/65 ${mono ? "font-mono" : ""}`} title={value}>{value}</span>
    </div>
  );
}

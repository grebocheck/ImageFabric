import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { CodeFile, CodeFileContent, Model } from "../types";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-emerald-500";
const CODE_SYSTEM = "You are a focused local code assistant. Prefer concrete file-aware answers. If context is missing, name exactly what is missing.";

function size(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1_000_000) return `${(bytes / 1000).toFixed(1)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function buildPrompt(prompt: string, files: CodeFileContent[]): string {
  const context = files.map((f) => (
    `<file path="${f.path}"${f.truncated ? ' truncated="true"' : ""}>\n${f.content}\n</file>`
  )).join("\n\n");
  return `${context ? `Repository context:\n\n${context}\n\n` : ""}Task:\n\n${prompt.trim()}`;
}

export function CodePanel({
  models,
  onOpenChat,
}: {
  models: Model[];
  onOpenChat: (conversationId: string, jobId: string) => void;
}) {
  const llmModels = useMemo(() => models.filter((m) => m.job_type === "llm"), [models]);
  const [modelId, setModelId] = useState("");
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<CodeFile[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [contents, setContents] = useState<Record<string, CodeFileContent>>({});
  const [activePath, setActivePath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  useEffect(() => {
    if (!modelId && llmModels[0]) setModelId(llmModels[0].id);
  }, [llmModels, modelId]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      api.listCodeFiles(query).then(setFiles).catch(() => setFiles([]));
    }, 150);
    return () => window.clearTimeout(t);
  }, [query]);

  async function loadFile(path: string): Promise<CodeFileContent | null> {
    if (contents[path]) return contents[path];
    try {
      const file = await api.getCodeFile(path);
      setContents((prev) => ({ ...prev, [path]: file }));
      return file;
    } catch (err) {
      setNote(err instanceof Error ? err.message : "could not read file");
      return null;
    }
  }

  async function toggle(path: string) {
    setActivePath(path);
    if (selected.includes(path)) {
      setSelected((prev) => prev.filter((p) => p !== path));
      return;
    }
    const file = await loadFile(path);
    if (file) setSelected((prev) => [...prev, path].slice(-8));
  }

  async function send() {
    if (!modelId || !prompt.trim() || busy) return;
    setBusy(true);
    setNote("");
    try {
      const loaded = (await Promise.all(selected.map(loadFile))).filter((f): f is CodeFileContent => Boolean(f));
      const conv = await api.createConversation({
        title: `Code: ${prompt.trim().slice(0, 80)}`,
        model_id: modelId,
        system: CODE_SYSTEM,
        params: { temperature: 0.2, max_tokens: 1400 },
      });
      const res = await api.sendChatMessage(conv.id, {
        content: buildPrompt(prompt, loaded),
        model_id: modelId,
        system: CODE_SYSTEM,
        temperature: 0.2,
        max_tokens: 1400,
      });
      onOpenChat(res.conversation.id, res.job_id);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  const active = activePath ? contents[activePath] : null;

  return (
    <div className="grid h-full grid-cols-[320px_1fr_360px] gap-3">
      <aside className="flex min-h-0 flex-col rounded-lg border border-white/10">
        <div className="border-b border-white/10 p-3">
          <div className="mb-2 text-sm font-semibold text-white/75">Repository files</div>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search paths"
            className={field}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {files.map((f) => {
            const isSelected = selected.includes(f.path);
            return (
              <button
                key={f.path}
                onClick={() => void toggle(f.path)}
                className={`mb-1 grid w-full grid-cols-[22px_1fr_auto] items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs ${
                  isSelected ? "bg-emerald-500/15 text-emerald-100" : "text-white/65 hover:bg-white/5"
                }`}
              >
                <span className="text-center text-white/45">{isSelected ? "-" : "+"}</span>
                <span className="min-w-0 truncate font-mono" title={f.path}>{f.path}</span>
                <span className="text-white/30">{size(f.size_bytes)}</span>
              </button>
            );
          })}
        </div>
      </aside>

      <section className="flex min-h-0 flex-col rounded-lg border border-white/10">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <span className="min-w-0 truncate font-mono text-sm text-white/60">
            {active?.path || selected[0] || "No file selected"}
          </span>
          {active?.truncated && <span className="text-xs text-amber-300/80">truncated</span>}
        </div>
        <pre className="min-h-0 flex-1 overflow-auto p-4 text-xs leading-5 text-white/70">
          {active?.content || ""}
        </pre>
      </section>

      <aside className="flex min-h-0 flex-col gap-3 rounded-lg border border-white/10 p-4">
        <label>
          <div className="text-xs uppercase tracking-wide text-white/40">Model</div>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={`${field} mt-1`}>
            {llmModels.length === 0 && <option value="">no LLM models</option>}
            {llmModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>

        <div>
          <div className="mb-1 text-xs uppercase tracking-wide text-white/40">Selected</div>
          <div className="max-h-32 overflow-y-auto rounded-md border border-white/10 bg-black/20 p-2">
            {selected.length === 0 ? (
              <div className="text-xs text-white/30">none</div>
            ) : selected.map((path) => (
              <button
                key={path}
                onClick={() => setActivePath(path)}
                className="block w-full truncate rounded px-1.5 py-1 text-left font-mono text-xs text-white/60 hover:bg-white/5"
                title={path}
              >
                {path}
              </button>
            ))}
          </div>
        </div>

        <label className="flex min-h-0 flex-1 flex-col">
          <div className="text-xs uppercase tracking-wide text-white/40">Prompt</div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className={`${field} mt-1 min-h-0 flex-1 resize-none`}
            placeholder="Task"
          />
        </label>

        <div className="flex items-center justify-between gap-3">
          <span className="min-w-0 truncate text-xs text-white/35" title={note || undefined}>{note}</span>
          <button
            onClick={() => void send()}
            disabled={!modelId || !prompt.trim() || busy}
            className="shrink-0 rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-30"
          >
            {busy ? "Sending..." : "Send to LLM"}
          </button>
        </div>
      </aside>
    </div>
  );
}

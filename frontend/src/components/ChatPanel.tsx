import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useEvents } from "../api/useEvents";
import type { BusEvent, ChatConversation, ChatConversationImport, ChatImportMessage, ChatMessage, ChatSendBody, LlmConfig, Model, Preset, PresetImportItem } from "../types";
import { Select } from "./Select";
import { AssistantContent } from "./Thinking";
import { Toggle } from "./Toggle";

const field = "w-full rounded-md bg-black/30 border border-white/10 px-2.5 py-1.5 text-sm outline-none focus:border-emerald-500";
const numField = "w-full rounded-md bg-black/30 border border-white/10 px-2 py-1 text-xs outline-none focus:border-emerald-500";
const label = "text-xs uppercase tracking-wide text-white/40";
const DEFAULTS_KEY = "hfabric.chat.defaults";

type NumOrEmpty = number | "";
type Stats = { tokens: number; tps: number; ttft: number };

function loadDefaults(): { model_id?: string; temperature?: number; max_tokens?: number } {
  try {
    return JSON.parse(localStorage.getItem(DEFAULTS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

const numOrUndef = (v: NumOrEmpty): number | undefined => (v === "" ? undefined : Number(v));
const parseStop = (s: string): string[] | undefined => {
  const items = s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean);
  return items.length ? items : undefined;
};

function pickImageModel(models: Model[]): Model | undefined {
  const img = models.filter((m) => m.job_type === "image");
  return img.find((m) => m.family === "flux2")
    ?? img.find((m) => m.quant?.startsWith("nunchaku"))
    ?? img.find((m) => !m.slow)
    ?? img[0];
}

type ImportBundle = { conversations: ChatConversationImport[]; presets: PresetImportItem[] };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function stringOrNull(v: unknown): string | null | undefined {
  return typeof v === "string" ? v : v === null ? null : undefined;
}

function asRecord(v: unknown): Record<string, unknown> {
  return isRecord(v) ? v : {};
}

function isPresent<T>(v: T | null | undefined): v is T {
  return v != null;
}

function asMessageImport(v: unknown): ChatImportMessage | null {
  if (!isRecord(v)) return null;
  const role = v.role;
  if (role !== "user" && role !== "assistant" && role !== "system") return null;
  return {
    role,
    content: typeof v.content === "string" ? v.content : "",
    error: typeof v.error === "boolean" ? v.error : undefined,
    created_at: typeof v.created_at === "string" ? v.created_at : undefined,
  };
}

function asConversationImport(v: unknown): ChatConversationImport | null {
  if (!isRecord(v)) return null;
  const hasConversationShape = Array.isArray(v.messages)
    || typeof v.title === "string"
    || typeof v.system === "string"
    || typeof v.model_id === "string";
  if (!hasConversationShape) return null;
  return {
    title: typeof v.title === "string" ? v.title : undefined,
    model_id: stringOrNull(v.model_id),
    system: stringOrNull(v.system),
    params: asRecord(v.params),
    created_at: typeof v.created_at === "string" ? v.created_at : undefined,
    updated_at: typeof v.updated_at === "string" ? v.updated_at : undefined,
    messages: Array.isArray(v.messages) ? v.messages.map(asMessageImport).filter(isPresent) : [],
  };
}

function asPresetImport(v: unknown): PresetImportItem | null {
  if (!isRecord(v)) return null;
  if (typeof v.name !== "string" || (v.type !== "llm" && v.type !== "image")) return null;
  return { name: v.name, type: v.type, params: asRecord(v.params) };
}

function parseImportBundle(data: unknown): ImportBundle {
  if (Array.isArray(data)) {
    return {
      conversations: data.map(asConversationImport).filter(isPresent),
      presets: data.map(asPresetImport).filter(isPresent),
    };
  }

  if (!isRecord(data)) return { conversations: [], presets: [] };

  const conversations = Array.isArray(data.conversations)
    ? data.conversations.map(asConversationImport).filter(isPresent)
    : (Array.isArray(data.messages) ? [asConversationImport(data)].filter(isPresent) : []);
  const presets = Array.isArray(data.presets)
    ? data.presets.map(asPresetImport).filter(isPresent)
    : [asPresetImport(data)].filter(isPresent);

  return { conversations, presets };
}

function downloadJson(filename: string, payload: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export type ChatJump = { conversationId: string; jobId?: string; nonce: number };

export function ChatPanel({ models, jump }: { models: Model[]; jump?: ChatJump | null }) {
  const llmModels = models.filter((m) => m.job_type === "llm");
  const saved = loadDefaults();

  const [convs, setConvs] = useState<ChatConversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [stats, setStats] = useState<Stats | null>(null);
  const [convQuery, setConvQuery] = useState("");

  // settings (per conversation)
  const [modelId, setModelId] = useState(saved.model_id ?? "");
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(saved.temperature ?? 0.8);
  const [maxTokens, setMaxTokens] = useState(saved.max_tokens ?? 512);
  // advanced sampling ("" = unset -> use model default)
  const [topP, setTopP] = useState<NumOrEmpty>("");
  const [topK, setTopK] = useState<NumOrEmpty>("");
  const [minP, setMinP] = useState<NumOrEmpty>("");
  const [repeatPenalty, setRepeatPenalty] = useState<NumOrEmpty>("");
  const [seed, setSeed] = useState<NumOrEmpty>("");
  const [stop, setStop] = useState("");
  const [imageTool, setImageTool] = useState(false);
  const [documentTool, setDocumentTool] = useState(false);
  const [ragTopK, setRagTopK] = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // personas (stored as llm presets)
  const [personas, setPersonas] = useState<Preset[]>([]);
  const [personaId, setPersonaId] = useState("");
  const [personaName, setPersonaName] = useState("");

  const [cfg, setCfg] = useState<LlmConfig | null>(null);
  const [ctxDraft, setCtxDraft] = useState<number | null>(null);
  const [cfgNote, setCfgNote] = useState("");
  const [importNote, setImportNote] = useState("");

  const activeJob = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  // streaming-stat trackers
  const sendStart = useRef(0);
  const firstAt = useRef<number | null>(null);
  const tokCount = useRef(0);

  const refreshConvs = useCallback(() => api.listConversations().then(setConvs).catch(() => {}), []);
  const refreshPersonas = useCallback(
    () => api.listPresets().then((p) => setPersonas(p.filter((x) => x.type === "llm"))).catch(() => {}),
    [],
  );

  useEffect(() => {
    refreshConvs();
    refreshPersonas();
    api.getLlmConfig().then((c) => { setCfg(c); setCtxDraft((p) => p ?? c.ctx); }).catch(() => {});
  }, [refreshConvs, refreshPersonas]);

  useEffect(() => {
    if (!modelId && llmModels[0]) setModelId(llmModels[0].id);
  }, [llmModels, modelId]);

  useEffect(() => {
    localStorage.setItem(DEFAULTS_KEY, JSON.stringify({ model_id: modelId, temperature, max_tokens: maxTokens }));
  }, [modelId, temperature, maxTokens]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // auto-grow the composer up to a cap, then scroll inside it
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  // --- live streaming + stats for the in-flight assistant message ---
  const onChatEvent = useCallback((e: BusEvent) => {
    if (e.job_id !== activeJob.current) return;
    if (e.type === "llm.token") {
      if (firstAt.current === null) firstAt.current = Date.now();
      tokCount.current += 1;
      setMessages((p) => appendToLastAssistant(p, e.token as string));
    } else if (e.type === "job.done") {
      const childJob = typeof e.tool_child_job_id === "string" ? e.tool_child_job_id : null;
      if (childJob) {
        activeJob.current = childJob;
        setBusy(true);
        if (typeof e.text === "string") setMessages((p) => setLastAssistant(p, e.text as string));
        return;
      }
      activeJob.current = null;
      setBusy(false);
      if (typeof e.text === "string") setMessages((p) => setLastAssistant(p, e.text as string));
      if (firstAt.current && tokCount.current > 0) {
        const secs = Math.max(0.001, (Date.now() - firstAt.current) / 1000);
        setStats({ tokens: tokCount.current, tps: tokCount.current / secs, ttft: firstAt.current - sendStart.current });
      }
      refreshConvs();
    } else if (e.type === "job.error") {
      activeJob.current = null;
      setBusy(false);
      setMessages((p) => setLastAssistant(p, `⚠ ${(e.error as string) ?? "generation failed"}`, true));
    } else if (e.type === "job.progress") {
      // image jobs (the /image bridge) report progress but stream no tokens
      const pct = Math.round(((e.progress as number) ?? 0) * 100);
      setMessages((p) => setLastAssistant(p, `*generating image… ${pct}%*`));
    } else if (e.type === "job.cancelled") {
      activeJob.current = null;
      setBusy(false);
    }
  }, [refreshConvs]);
  useEvents(onChatEvent);

  const selectConversation = useCallback(async (id: string) => {
    setActiveId(id);
    setEditingId(null);
    setStats(null);
    try {
      const d = await api.getConversation(id);
      setMessages(d.messages);
      if (d.model_id) setModelId(d.model_id);
      setSystem(d.system ?? "");
      const pr = d.params ?? {};
      if (typeof pr.temperature === "number") setTemperature(pr.temperature);
      if (typeof pr.max_tokens === "number") setMaxTokens(pr.max_tokens);
      setTopP(typeof pr.top_p === "number" ? pr.top_p : "");
      setTopK(typeof pr.top_k === "number" ? pr.top_k : "");
      setMinP(typeof pr.min_p === "number" ? pr.min_p : "");
      setRepeatPenalty(typeof pr.repeat_penalty === "number" ? pr.repeat_penalty : "");
      setStop(Array.isArray(pr.stop) ? (pr.stop as string[]).join(", ") : "");
      setImageTool(Boolean(pr.image_tool));
      setDocumentTool(Boolean(pr.document_tool));
      setRagTopK(typeof pr.rag_top_k === "number" ? pr.rag_top_k : 5);
    } catch {
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    if (!activeId && convs[0]) void selectConversation(convs[0].id);
  }, [convs, activeId, selectConversation]);

  useEffect(() => {
    if (!jump?.conversationId) return;
    activeJob.current = jump.jobId ?? null;
    if (jump.jobId) {
      setBusy(true);
      sendStart.current = Date.now();
      firstAt.current = null;
      tokCount.current = 0;
    }
    refreshConvs();
    void selectConversation(jump.conversationId);
  }, [jump, refreshConvs, selectConversation]);

  const newChat = useCallback(async () => {
    const c = await api.createConversation({ model_id: modelId || llmModels[0]?.id });
    setConvs((p) => [c, ...p]);
    setActiveId(c.id);
    setMessages([]);
    setEditingId(null);
    setStats(null);
  }, [modelId, llmModels]);

  const deleteConversation = useCallback(async (id: string) => {
    await api.deleteConversation(id).catch(() => {});
    setConvs((p) => p.filter((c) => c.id !== id));
    if (activeId === id) { setActiveId(null); setMessages([]); }
  }, [activeId]);

  const sampling = useCallback((): Omit<ChatSendBody, "content" | "model_id"> => ({
    system: system.trim() || undefined,
    temperature,
    max_tokens: maxTokens,
    top_p: numOrUndef(topP),
    top_k: numOrUndef(topK),
    min_p: numOrUndef(minP),
    repeat_penalty: numOrUndef(repeatPenalty),
    seed: numOrUndef(seed),
    stop: parseStop(stop),
  }), [system, temperature, maxTokens, topP, topK, minP, repeatPenalty, seed, stop]);

  const submit = useCallback(async (content: string, convId: string) => {
    const mdl = modelId || llmModels[0]?.id;
    if (!mdl) return;
    setBusy(true);
    setStats(null);
    sendStart.current = Date.now();
    firstAt.current = null;
    tokCount.current = 0;
    setMessages((p) => [...p, { id: "tmp-u", role: "user", content }, { id: "tmp-a", role: "assistant", content: "" }]);
    try {
      const img = imageTool ? pickImageModel(models) : undefined;
      const res = await api.sendChatMessage(convId, {
        content,
        model_id: mdl,
        ...sampling(),
        ...(imageTool && img ? { image_tool: true, image_model_id: img.id } : {}),
        ...(documentTool ? { document_tool: true, rag_top_k: ragTopK } : {}),
      });
      activeJob.current = res.job_id;
      setMessages((p) => p.map((m) =>
        m.id === "tmp-u" ? res.user_message : m.id === "tmp-a" ? { ...res.assistant_message, content: "" } : m,
      ));
      setConvs((p) => [res.conversation, ...p.filter((c) => c.id !== res.conversation.id)]);
    } catch (err) {
      activeJob.current = null;
      setBusy(false);
      setMessages((p) => setLastAssistant(p, `⚠ ${err instanceof Error ? err.message : "request failed"}`, true));
    }
  }, [documentTool, imageTool, modelId, llmModels, models, ragTopK, sampling]);

  const submitImage = useCallback(async (prompt: string, convId: string) => {
    const img = pickImageModel(models);
    if (!img) {
      setMessages((p) => [...p, { id: "tmp-u", role: "user", content: `/image ${prompt}` },
        { id: "tmp-a", role: "assistant", content: "⚠ no image model available", error: true }]);
      return;
    }
    setBusy(true);
    setStats(null);
    setMessages((p) => [...p, { id: "tmp-u", role: "user", content: `/image ${prompt}` },
      { id: "tmp-a", role: "assistant", content: "" }]);
    try {
      const res = await api.sendChatImage(convId, { prompt, model_id: img.id });
      activeJob.current = res.job_id;
      setMessages((p) => p.map((m) =>
        m.id === "tmp-u" ? res.user_message : m.id === "tmp-a" ? { ...res.assistant_message, content: "" } : m,
      ));
      setConvs((p) => [res.conversation, ...p.filter((c) => c.id !== res.conversation.id)]);
    } catch (err) {
      activeJob.current = null;
      setBusy(false);
      setMessages((p) => setLastAssistant(p, `⚠ ${err instanceof Error ? err.message : "request failed"}`, true));
    }
  }, [models]);

  const send = useCallback(async () => {
    const content = input.trim();
    if (!content || busy) return;
    let cid = activeId;
    if (!cid) {
      const c = await api.createConversation({ model_id: modelId || llmModels[0]?.id });
      setConvs((p) => [c, ...p]);
      setActiveId(c.id);
      cid = c.id;
    }
    setInput("");
    const imgCmd = content.match(/^\/(?:image|img)\s+([\s\S]+)/i);
    if (imgCmd) await submitImage(imgCmd[1].trim(), cid);
    else await submit(content, cid);
  }, [input, busy, activeId, modelId, llmModels, submit, submitImage]);

  const stop_ = useCallback(async () => {
    await api.stopLlm().catch(() => {});
    if (activeJob.current) await api.cancelJob(activeJob.current).catch(() => {});
  }, []);

  const regenerate = useCallback(async () => {
    if (busy || !activeId) return;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser || lastUser.id.startsWith("tmp")) return;
    await api.truncateFrom(activeId, lastUser.id).catch(() => {});
    const idx = messages.findIndex((m) => m.id === lastUser.id);
    setMessages((p) => p.slice(0, idx));
    await submit(lastUser.content, activeId);
  }, [busy, activeId, messages, submit]);

  const startEdit = (m: ChatMessage) => { setEditingId(m.id); setEditText(m.content); };
  const saveEdit = useCallback(async () => {
    if (!activeId || !editingId) return;
    const content = editText.trim();
    const idx = messages.findIndex((m) => m.id === editingId);
    setEditingId(null);
    if (!content || idx < 0) return;
    await api.truncateFrom(activeId, editingId).catch(() => {});
    setMessages((p) => p.slice(0, idx));
    await submit(content, activeId);
  }, [activeId, editingId, editText, messages, submit]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
  };

  const applyCtx = async () => {
    if (ctxDraft == null) return;
    setCfgNote("");
    try {
      const next = await api.setLlmConfig({ ctx: ctxDraft });
      setCfg(next); setCtxDraft(next.ctx);
      setCfgNote(next.reloaded ? "applied — model reloaded" : next.changed ? "applied (next load)" : "no change");
    } catch (err) {
      setCfgNote(err instanceof Error ? err.message : "could not update");
    }
  };

  // --- personas ---
  const applyPersona = (id: string) => {
    setPersonaId(id);
    const p = personas.find((x) => x.id === id);
    if (!p) return;
    const pr = p.params ?? {};
    setSystem(typeof pr.system === "string" ? pr.system : "");
    if (typeof pr.temperature === "number") setTemperature(pr.temperature);
    if (typeof pr.max_tokens === "number") setMaxTokens(pr.max_tokens);
    setTopP(typeof pr.top_p === "number" ? pr.top_p : "");
    setTopK(typeof pr.top_k === "number" ? pr.top_k : "");
    setMinP(typeof pr.min_p === "number" ? pr.min_p : "");
    setRepeatPenalty(typeof pr.repeat_penalty === "number" ? pr.repeat_penalty : "");
    setStop(Array.isArray(pr.stop) ? (pr.stop as string[]).join(", ") : "");
  };

  const savePersona = async () => {
    const name = personaName.trim();
    if (!name) return;
    const s = sampling();
    await api.createPreset(name, "llm", {
      system: system.trim(),
      temperature, max_tokens: maxTokens,
      ...(s.top_p !== undefined ? { top_p: s.top_p } : {}),
      ...(s.top_k !== undefined ? { top_k: s.top_k } : {}),
      ...(s.min_p !== undefined ? { min_p: s.min_p } : {}),
      ...(s.repeat_penalty !== undefined ? { repeat_penalty: s.repeat_penalty } : {}),
      ...(s.stop ? { stop: s.stop } : {}),
    }).catch(() => {});
    setPersonaName("");
    refreshPersonas();
  };

  const deletePersona = async () => {
    if (!personaId) return;
    await api.deletePreset(personaId).catch(() => {});
    setPersonaId("");
    refreshPersonas();
  };

  const exportChat = () => {
    if (!messages.length) return;
    const title = convs.find((c) => c.id === activeId)?.title ?? "chat";
    const md = `# ${title}\n\n` + messages
      .map((m) => `**${m.role}:**\n\n${m.content}\n`)
      .join("\n---\n\n");
    const url = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.slice(0, 40).replace(/[^a-z0-9]+/gi, "-") || "chat"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportJson = () => {
    const activeConv = convs.find((c) => c.id === activeId);
    const conversations = activeConv ? [{
      title: activeConv.title,
      model_id: activeConv.model_id,
      system: activeConv.system,
      params: activeConv.params,
      created_at: activeConv.created_at,
      updated_at: activeConv.updated_at,
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
        error: m.error,
        created_at: m.created_at,
      })),
    }] : [];
    const presets = personas.map((p) => ({ name: p.name, type: p.type, params: p.params }));
    const title = activeConv?.title ?? "chat";
    const slug = title.slice(0, 40).replace(/[^a-z0-9]+/gi, "-") || "hfabric";
    downloadJson(`${slug}.hfabric.json`, {
      format: "hfabric.bundle.v1",
      exported_at: new Date().toISOString(),
      conversations,
      presets,
    });
  };

  const importJson = useCallback(async (file: File | null) => {
    if (!file) return;
    setImportNote("");
    try {
      const bundle = parseImportBundle(JSON.parse(await file.text()));
      const parts: string[] = [];
      let firstImportedConversation: string | null = null;

      if (bundle.conversations.length) {
        const res = await api.importConversations(bundle.conversations);
        firstImportedConversation = res.conversations[0]?.id ?? null;
        parts.push(`${res.imported} chat${res.imported === 1 ? "" : "s"}`);
      }

      if (bundle.presets.length) {
        const res = await api.importPresets(bundle.presets, "rename");
        parts.push(`${res.imported} preset${res.imported === 1 ? "" : "s"}`);
      }

      if (!parts.length) {
        setImportNote("nothing importable in file");
        return;
      }

      await refreshConvs();
      await refreshPersonas();
      if (firstImportedConversation) await selectConversation(firstImportedConversation);
      setImportNote(`imported ${parts.join(", ")}`);
    } catch (err) {
      setImportNote(err instanceof Error ? err.message : "import failed");
    } finally {
      if (importInputRef.current) importInputRef.current.value = "";
    }
  }, [refreshConvs, refreshPersonas, selectConversation]);

  const filteredConvs = convQuery.trim()
    ? convs.filter((c) => c.title.toLowerCase().includes(convQuery.trim().toLowerCase()))
    : convs;

  const approxTokens = Math.ceil(
    (system.length + input.length + messages.reduce((n, m) => n + m.content.length, 0)) / 4,
  );

  return (
    <div className="flex h-full gap-3">
      {/* --- conversations --- */}
      <aside className="flex w-56 shrink-0 flex-col rounded-lg border border-white/10">
        <button onClick={() => void newChat()} className="mx-2 mt-2 rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium hover:bg-emerald-500">
          + New chat
        </button>
        <input
          value={convQuery}
          onChange={(e) => setConvQuery(e.target.value)}
          placeholder="search chats"
          className="mx-2 my-2 rounded-md border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none focus:border-emerald-500"
        />
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {filteredConvs.length === 0 && <div className="px-1 text-xs text-white/30">no conversations</div>}
          {filteredConvs.map((c) => (
            <div
              key={c.id}
              onClick={() => void selectConversation(c.id)}
              className={`group mb-1 flex cursor-pointer items-center justify-between gap-1 rounded-md px-2 py-1.5 text-sm ${
                activeId === c.id ? "bg-white/15" : "hover:bg-white/5"
              }`}
            >
              <span className="min-w-0 flex-1 truncate text-white/80">{c.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); void deleteConversation(c.id); }}
                className="shrink-0 text-white/30 opacity-0 transition hover:text-red-300 group-hover:opacity-100"
                title="delete"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* --- conversation --- */}
      <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-white/10">
        <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-center text-sm text-white/30">
              Start a conversation with the local model.
            </div>
          ) : (
            messages.map((m) => (
              <Bubble
                key={m.id}
                msg={m}
                editing={editingId === m.id}
                editText={editText}
                setEditText={setEditText}
                onStartEdit={() => startEdit(m)}
                onSaveEdit={() => void saveEdit()}
                onCancelEdit={() => setEditingId(null)}
              />
            ))
          )}
        </div>

        <div className="border-t border-white/10 p-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            placeholder={modelId ? "Message…  (Enter to send, Shift+Enter for newline)" : "no LLM model available"}
            disabled={!modelId}
            className={`${field} max-h-[200px] resize-none`}
          />
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-white/35">
              ~{approxTokens} / {cfg?.ctx ?? "?"} tokens
              <span className="ml-2 text-white/25">· /image &lt;prompt&gt; to generate</span>
              {imageTool && <span className="ml-2 text-white/25">· image tool on</span>}
              {documentTool && <span className="ml-2 text-white/25">· document tool on</span>}
              {stats && <span className="ml-2 text-white/30">· {stats.tps.toFixed(1)} tok/s · TTFT {Math.round(stats.ttft)}ms</span>}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => void regenerate()}
                disabled={busy || !messages.some((m) => m.role === "assistant")}
                className="rounded-md border border-white/15 px-2.5 py-1.5 text-xs hover:bg-white/10 disabled:opacity-30"
              >
                Regenerate
              </button>
              {busy ? (
                <button onClick={() => void stop_()} className="rounded-md border border-red-400/40 px-4 py-1.5 text-sm font-medium text-red-200 hover:bg-red-400/10">
                  Stop
                </button>
              ) : (
                <button
                  onClick={() => void send()}
                  disabled={!input.trim() || !modelId}
                  className="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-40"
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* --- settings --- */}
      <aside className="flex w-72 shrink-0 flex-col gap-4 overflow-y-auto rounded-lg border border-white/10 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white/75">Model settings</h2>
          <div className="flex gap-1">
            <button
              onClick={exportChat}
              disabled={!messages.length}
              className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
              title="Export conversation as Markdown"
            >
              MD
            </button>
            <button
              onClick={exportJson}
              disabled={!activeId && personas.length === 0}
              className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10 disabled:opacity-30"
              title="Export importable JSON bundle"
            >
              JSON
            </button>
            <button
              onClick={() => importInputRef.current?.click()}
              className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10"
              title="Import JSON bundle"
            >
              Import
            </button>
            <input
              ref={importInputRef}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(e) => void importJson(e.currentTarget.files?.[0] ?? null)}
            />
          </div>
        </div>
        {importNote && <div className="text-[11px] text-emerald-300/80">{importNote}</div>}

        <label>
          <div className={label}>Model</div>
          <Select
            value={modelId}
            onChange={setModelId}
            placeholder="no LLM models"
            className="mt-1"
            options={llmModels.map((m) => ({ value: m.id, label: m.name }))}
          />
        </label>

        <div className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2">
          <span>
            <span className="block text-sm font-medium text-white/70">Image tool</span>
            <span className="block text-xs text-white/35">{pickImageModel(models)?.name ?? "no image model"}</span>
          </span>
          <Toggle checked={imageTool} disabled={!pickImageModel(models)} onChange={setImageTool} />
        </div>

        <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
          <div className="flex items-center justify-between gap-3">
            <span>
              <span className="block text-sm font-medium text-white/70">Document tool</span>
              <span className="block text-xs text-white/35">model-driven RAG search</span>
            </span>
            <Toggle checked={documentTool} onChange={setDocumentTool} />
          </div>
          {documentTool && (
            <label className="mt-2 block">
              <div className={label}>RAG top K</div>
              <input
                type="number"
                min={1}
                max={20}
                value={ragTopK}
                onChange={(e) => setRagTopK(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
                className={`${numField} mt-1`}
              />
            </label>
          )}
        </div>

        <div>
          <div className={label}>Context window (tokens)</div>
          <div className="mt-1 flex gap-2">
            <input type="number" min={512} step={512} value={ctxDraft ?? ""} onChange={(e) => setCtxDraft(Number(e.target.value))} className={numField} />
            <button onClick={() => void applyCtx()} disabled={ctxDraft == null || ctxDraft === cfg?.ctx}
              className="shrink-0 rounded-md border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30">
              Apply
            </button>
          </div>
          <div className="mt-1 text-[11px] text-white/35">current {cfg?.ctx ?? "?"} · {cfg?.loaded ? "loaded" : "not loaded"}</div>
          {cfgNote && <div className="mt-1 text-[11px] text-emerald-300/80">{cfgNote}</div>}
          {ctxDraft != null && cfg && ctxDraft !== cfg.ctx && cfg.loaded && (
            <div className="mt-1 text-[11px] text-amber-300/80">applying reloads the running model</div>
          )}
        </div>

        <label>
          <div className={label}>Temperature · {temperature.toFixed(2)}</div>
          <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="mt-2 w-full accent-emerald-500" />
        </label>

        <label>
          <div className={label}>Max tokens</div>
          <input type="number" min={1} max={8192} step={64} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} className={`${numField} mt-1`} />
        </label>

        {/* advanced sampling */}
        <div>
          <button onClick={() => setShowAdvanced((v) => !v)} className="flex w-full items-center justify-between text-xs uppercase tracking-wide text-white/40 hover:text-white/70">
            <span>Advanced sampling</span>
            <span>{showAdvanced ? "▾" : "▸"}</span>
          </button>
          {showAdvanced && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              <NumOpt label="top_p" v={topP} set={setTopP} step={0.05} />
              <NumOpt label="top_k" v={topK} set={setTopK} step={1} />
              <NumOpt label="min_p" v={minP} set={setMinP} step={0.01} />
              <NumOpt label="repeat_pen" v={repeatPenalty} set={setRepeatPenalty} step={0.05} />
              <NumOpt label="seed" v={seed} set={setSeed} step={1} />
              <label className="col-span-2">
                <div className={label}>stop (comma-sep)</div>
                <input value={stop} onChange={(e) => setStop(e.target.value)} placeholder="empty = none" className={`${numField} mt-1`} />
              </label>
            </div>
          )}
        </div>

        {/* persona */}
        <div>
          <div className={label}>Persona</div>
          <div className="mt-1 grid grid-cols-[1fr_auto] gap-2">
            <Select
              value={personaId}
              onChange={applyPersona}
              placeholder="— none —"
              options={[{ value: "", label: "— none —" }, ...personas.map((p) => ({ value: p.id, label: p.name }))]}
            />
            <button onClick={() => void deletePersona()} disabled={!personaId}
              className="rounded-md border border-red-400/25 px-2 py-1 text-xs text-red-300 hover:bg-red-400/10 disabled:opacity-30">
              Del
            </button>
          </div>
          <div className="mt-1 grid grid-cols-[1fr_auto] gap-2">
            <input value={personaName} onChange={(e) => setPersonaName(e.target.value)} placeholder="save current as…" className={numField} />
            <button onClick={() => void savePersona()} disabled={!personaName.trim()}
              className="rounded-md border border-white/15 px-2.5 py-1 text-xs hover:bg-white/10 disabled:opacity-30">
              Save
            </button>
          </div>
        </div>

        <label className="flex min-h-0 flex-1 flex-col">
          <div className={label}>System prompt</div>
          <textarea
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            placeholder="optional — sets the assistant's behavior"
            className={`${field} mt-1 min-h-24 flex-1 resize-none`}
          />
        </label>
      </aside>
    </div>
  );
}

function NumOpt({ label: l, v, set, step }: { label: string; v: NumOrEmpty; set: (n: NumOrEmpty) => void; step: number }) {
  return (
    <label>
      <div className={label}>{l}</div>
      <input
        type="number"
        step={step}
        value={v}
        onChange={(e) => set(e.target.value === "" ? "" : Number(e.target.value))}
        placeholder="default"
        className={`${numField} mt-1`}
      />
    </label>
  );
}

function Bubble({
  msg, editing, editText, setEditText, onStartEdit, onSaveEdit, onCancelEdit,
}: {
  msg: ChatMessage;
  editing: boolean;
  editText: string;
  setEditText: (v: string) => void;
  onStartEdit: () => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(msg.content).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  };

  if (editing) {
    return (
      <div className="flex justify-end">
        <div className="w-[80%]">
          <textarea value={editText} onChange={(e) => setEditText(e.target.value)} rows={3} className={`${field} resize-none`} />
          <div className="mt-1 flex justify-end gap-2">
            <button onClick={onCancelEdit} className="rounded border border-white/15 px-2 py-1 text-xs hover:bg-white/10">Cancel</button>
            <button onClick={onSaveEdit} className="rounded bg-emerald-600 px-2.5 py-1 text-xs font-medium hover:bg-emerald-500">Save &amp; resend</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`group flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] rounded-lg px-3 py-2 ${
        isUser ? "bg-violet-600/30 text-white"
          : msg.error ? "border border-red-400/30 bg-red-400/10 text-red-200"
          : "border border-white/10 bg-white/[0.04]"
      }`}>
        {isUser ? (
          <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
        ) : (
          <AssistantContent content={msg.content} />
        )}
        <div className="mt-1 flex gap-2 opacity-0 transition group-hover:opacity-100">
          <button onClick={copy} className="text-[11px] text-white/40 hover:text-white/80">{copied ? "copied" : "copy"}</button>
          {isUser && !msg.id.startsWith("tmp") && (
            <button onClick={onStartEdit} className="text-[11px] text-white/40 hover:text-white/80">edit</button>
          )}
        </div>
      </div>
    </div>
  );
}

function appendToLastAssistant(msgs: ChatMessage[], token: string): ChatMessage[] {
  const out = [...msgs];
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i].role === "assistant") { out[i] = { ...out[i], content: out[i].content + token }; return out; }
  }
  return out;
}

function setLastAssistant(msgs: ChatMessage[], content: string, error = false): ChatMessage[] {
  const out = [...msgs];
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i].role === "assistant") { out[i] = { ...out[i], content, error }; return out; }
  }
  return out;
}

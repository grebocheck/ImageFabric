import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import type { VoiceSettingsUpdate, VoiceStatus } from "../types";

const field = "w-full rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5 text-sm outline-none focus:border-violet-500";
const f0Options = [
  { value: "rmvpe_onnx", label: "RMVPE ONNX" },
  { value: "rmvpe", label: "RMVPE" },
  { value: "crepe_onnx_tiny", label: "CREPE tiny ONNX" },
  { value: "crepe_onnx_full", label: "CREPE full ONNX" },
  { value: "crepe_tiny", label: "CREPE tiny" },
  { value: "crepe_full", label: "CREPE full" },
  { value: "fcpe", label: "FCPE" },
  { value: "fcpe_onnx", label: "FCPE ONNX" },
];

const sampleRates = [16000, 24000, 44100, 48000, 96000];

function size(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}

function num(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function selectedModelId(status: VoiceStatus): string {
  const slot = status.selected_model_slot;
  return status.models.find((m) => m.slot === slot)?.id ?? status.models[0]?.id ?? "";
}

function perfSummary(performance: Record<string, unknown> | null): string {
  if (!performance) return "...";
  const entries = Object.entries(performance)
    .filter(([, value]) => ["number", "string", "boolean"].includes(typeof value))
    .slice(0, 3)
    .map(([key, value]) => `${key}:${String(value)}`);
  return entries.join(", ") || "available";
}

function deviceHint(hostApi: string, rate: number | null): string {
  return [hostApi, rate ? `${rate / 1000}k` : ""].filter(Boolean).join(", ");
}

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [modelId, setModelId] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRatio, setIndexRatio] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Detector, setF0Detector] = useState("rmvpe_onnx");
  const [inputDeviceId, setInputDeviceId] = useState(-1);
  const [outputDeviceId, setOutputDeviceId] = useState(-1);
  const [monitorDeviceId, setMonitorDeviceId] = useState(-1);
  const [sampleRate, setSampleRate] = useState(48000);
  const [readChunkSize, setReadChunkSize] = useState(133);
  const [inputGain, setInputGain] = useState(1);
  const [outputGain, setOutputGain] = useState(1);
  const [monitorGain, setMonitorGain] = useState(1);

  const refresh = useCallback(async () => {
    try {
      const next = await api.voiceStatus();
      setStatus(next);
      setError("");
      return next;
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load status");
      return null;
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (!status) return;
    setModelId((prev) => selectedModelId(status) || prev);
    setPitch(num(status.settings.tran, 0));
    setIndexRatio(num(status.settings.indexRatio, 1));
    setProtect(num(status.settings.protect, 0.5));
    const f0 = String(status.settings.f0Detector ?? "rmvpe_onnx");
    setF0Detector(f0Options.some((o) => o.value === f0) ? f0 : "rmvpe_onnx");
    setInputDeviceId(num(status.settings.serverInputDeviceId, -1));
    setOutputDeviceId(num(status.settings.serverOutputDeviceId, -1));
    setMonitorDeviceId(num(status.settings.serverMonitorDeviceId, -1));
    setSampleRate(num(status.settings.serverAudioSampleRate, 48000));
    setReadChunkSize(num(status.settings.serverReadChunkSize, 133));
    setInputGain(num(status.settings.serverInputAudioGain, 1));
    setOutputGain(num(status.settings.serverOutputAudioGain, 1));
    setMonitorGain(num(status.settings.serverMonitorAudioGain, 1));
  }, [status]);

  const models = status?.models ?? [];
  const inputDevices = status?.audio_devices.inputs ?? [];
  const outputDevices = status?.audio_devices.outputs ?? [];
  const selected = useMemo(() => models.find((m) => m.id === modelId), [models, modelId]);
  const canReach = Boolean(status?.server_reachable);
  const canControl = canReach && models.length > 0 && !busy;
  const live = Boolean(status?.server_audio_enabled || status?.voice_lane_active);
  const streamStarted = Boolean(status?.server_audio_started);

  const body = (): VoiceSettingsUpdate => ({
    model_id: modelId || null,
    pitch,
    index_ratio: indexRatio,
    protect,
    f0_detector: f0Detector,
    server_input_device_id: inputDeviceId,
    server_output_device_id: outputDeviceId,
    server_monitor_device_id: monitorDeviceId,
    server_audio_sample_rate: sampleRate,
    server_read_chunk_size: readChunkSize,
    server_input_gain: inputGain,
    server_output_gain: outputGain,
    server_monitor_gain: monitorGain,
  });

  async function pollForServer() {
    for (let i = 0; i < 20; i += 1) {
      const next = await api.voiceStatus();
      setStatus(next);
      if (next.server_reachable) return next;
      await delay(1000);
    }
    return api.voiceStatus().then((next) => {
      setStatus(next);
      return next;
    });
  }

  async function run(label: string, fn: () => Promise<VoiceStatus | null | void>) {
    setBusy(label);
    setError("");
    try {
      const next = await fn();
      if (next) setStatus(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  }

  const onStartServer = () => run("start-server", async () => {
    await api.voiceStartServer();
    return pollForServer();
  });

  const onStopServer = () => run("stop-server", async () => {
    await api.voiceStopServer();
    return refresh();
  });

  const onApply = () => run("apply", () => api.voiceApplySettings(body()));

  const onLive = (next: boolean) => run(next ? "live-on" : "live-off", () => (
    next ? api.voiceStartSession(body()) : api.voiceStopSession()
  ));

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4 overflow-y-auto p-1">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white/85">Voice changer</h2>
          <p className="mt-1 text-sm text-white/45">
            w-okada / MMVCServerSIO <span className="text-white/25">|</span> {live ? "live voice lane active" : "voice lane idle"}
          </p>
        </div>
        <button
          onClick={() => void refresh()}
          disabled={Boolean(busy)}
          className="shrink-0 rounded-md border border-white/15 px-2.5 py-1.5 text-xs text-white/70 transition hover:bg-white/10 hover:text-white disabled:opacity-40"
        >
          Refresh
        </button>
      </header>

      {error ? (
        <div className="rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">{error}</div>
      ) : null}

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="text-xs font-medium uppercase tracking-wide text-white/40">Engine</div>
          <div className="flex items-center gap-2">
            <Badge color={status?.wokada_installed ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
              {status?.wokada_installed ? "installed" : "missing"}
            </Badge>
            <Badge color={canReach ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
              {canReach ? "reachable" : "offline"}
            </Badge>
            {status?.server_running ? <Badge color="bg-sky-700/50 text-sky-100">managed</Badge> : null}
          </div>
        </div>

        <div className="grid gap-2 text-sm md:grid-cols-2">
          <Row label="Executable" value={status?.executable ?? "not found"} ok={status?.wokada_installed} mono />
          <Row label="Server" value={status?.server_url ?? "..."} ok={canReach} mono />
          <Row label="Selected" value={selected?.name ?? (status?.selected_model_slot ? `slot ${status.selected_model_slot}` : "none")} />
          <Row label="Performance" value={perfSummary(status?.performance ?? null)} />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            onClick={onStartServer}
            disabled={!status?.wokada_installed || canReach || Boolean(busy)}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-30 disabled:hover:bg-emerald-600"
          >
            {busy === "start-server" ? "Starting..." : "Start server"}
          </button>
          <button
            onClick={onStopServer}
            disabled={!status?.server_running || Boolean(busy)}
            className="rounded-md border border-red-400/35 px-3 py-1.5 text-sm font-medium text-red-100 hover:bg-red-400/10 disabled:opacity-30"
          >
            {busy === "stop-server" ? "Stopping..." : "Stop server"}
          </button>
          {canReach ? (
            <a
              href={status?.server_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-white/15 px-3 py-1.5 text-sm text-white/75 transition hover:bg-white/10 hover:text-white"
            >
              Open w-okada UI
            </a>
          ) : null}
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-white/40">Live controls</div>
            <div className="mt-1 text-xs text-white/35">
              {canReach ? "server API ready" : "start MMVCServerSIO first"}
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-white/55">
            <span>{streamStarted ? "Stream on" : live ? "Live armed" : "Live off"}</span>
            <Toggle checked={live} onChange={onLive} disabled={!canControl && !live} />
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Voice</div>
            <Select
              value={modelId}
              onChange={setModelId}
              placeholder="no voices"
              className="mt-1"
              options={models.map((m) => ({ value: m.id, label: m.name, hint: `#${m.slot}` }))}
            />
          </label>

          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">F0 detector</div>
            <Select value={f0Detector} onChange={setF0Detector} className="mt-1" options={f0Options} />
          </label>

          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Pitch</div>
            <input
              type="number"
              min={-24}
              max={24}
              step={1}
              value={pitch}
              onChange={(e) => setPitch(Number(e.target.value))}
              className={`${field} mt-1`}
            />
          </label>

          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Index ratio</div>
            <Slider value={indexRatio} min={0} max={1} step={0.01} onChange={setIndexRatio} />
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Protect</div>
            <Slider value={protect} min={0} max={1} step={0.01} onChange={setProtect} />
          </div>

          <div className="flex items-end">
            <button
              onClick={onApply}
              disabled={!canControl}
              className="w-full rounded-md border border-white/15 px-3 py-1.5 text-sm font-medium text-white/75 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
            >
              {busy === "apply" ? "Applying..." : "Apply settings"}
            </button>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-white/40">Audio routing</div>
            <div className="mt-1 text-xs text-white/35">
              {canReach ? `${inputDevices.length} inputs / ${outputDevices.length} outputs` : "available after server start"}
            </div>
          </div>
          <Badge color={streamStarted ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
            {streamStarted ? "streaming" : "stopped"}
          </Badge>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Input</div>
            <Select
              value={String(inputDeviceId)}
              onChange={(v) => setInputDeviceId(Number(v))}
              placeholder="start server"
              className="mt-1"
              options={inputDevices.map((d) => ({
                value: d.id,
                label: d.name,
                hint: deviceHint(d.host_api, d.default_sample_rate),
              }))}
            />
          </label>

          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Output</div>
            <Select
              value={String(outputDeviceId)}
              onChange={(v) => setOutputDeviceId(Number(v))}
              placeholder="start server"
              className="mt-1"
              options={outputDevices.map((d) => ({
                value: d.id,
                label: d.name,
                hint: deviceHint(d.host_api, d.default_sample_rate),
              }))}
            />
          </label>

          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Monitor</div>
            <Select
              value={String(monitorDeviceId)}
              onChange={(v) => setMonitorDeviceId(Number(v))}
              className="mt-1"
              options={[
                { value: "-1", label: "none" },
                ...outputDevices.map((d) => ({
                  value: d.id,
                  label: d.name,
                  hint: deviceHint(d.host_api, d.default_sample_rate),
                })),
              ]}
            />
          </label>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Sample rate</div>
            <Select
              value={String(sampleRate)}
              onChange={(v) => setSampleRate(Number(v))}
              className="mt-1"
              options={sampleRates.map((rate) => ({ value: String(rate), label: `${rate} Hz` }))}
            />
          </label>

          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Chunk</div>
            <input
              type="number"
              min={1}
              max={1024}
              value={readChunkSize}
              onChange={(e) => setReadChunkSize(Number(e.target.value))}
              className={`${field} mt-1`}
            />
          </label>

          <div className="flex items-end">
            <button
              onClick={onApply}
              disabled={!canReach || Boolean(busy)}
              className="w-full rounded-md border border-white/15 px-3 py-1.5 text-sm font-medium text-white/75 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
            >
              {busy === "apply" ? "Applying..." : "Apply routing"}
            </button>
          </div>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Input gain</div>
            <Slider value={inputGain} min={0} max={2} step={0.01} onChange={setInputGain} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Output gain</div>
            <Slider value={outputGain} min={0} max={2} step={0.01} onChange={setOutputGain} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Monitor gain</div>
            <Slider value={monitorGain} min={0} max={2} step={0.01} onChange={setMonitorGain} />
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-xs font-medium uppercase tracking-wide text-white/40">Voices</div>
          <Badge>{models.length}</Badge>
        </div>
        {models.length === 0 ? (
          <p className="text-sm leading-6 text-white/40">
            No voice slots found in <code className="text-white/60">{status?.model_dir ?? "model_dir"}</code>.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {models.map((m) => (
              <li
                key={m.id}
                onClick={() => setModelId(m.id)}
                className={`flex cursor-pointer items-center justify-between gap-2 rounded-md border px-3 py-2 transition ${
                  m.id === modelId ? "border-violet-400/40 bg-violet-500/10" : "border-white/10 bg-black/20 hover:bg-white/5"
                }`}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className="text-[10px] text-white/30">#{m.slot}</span>
                  <span className="min-w-0 truncate text-sm text-white/80" title={m.name}>{m.name}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <Badge color="bg-violet-700/50 text-violet-100">{m.type}{m.version ? ` ${m.version}` : ""}</Badge>
                  {m.f0 ? <Badge color="bg-sky-700/50 text-sky-100">f0</Badge> : null}
                  {m.has_index ? <Badge color="bg-emerald-700/55 text-emerald-100">index</Badge> : <Badge>no index</Badge>}
                  <span className="font-mono text-xs text-white/35">{size(m.size_bytes)}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Row({ label, value, ok, mono = false }: { label: string; value: string; ok?: boolean; mono?: boolean }) {
  return (
    <div className="grid min-w-0 grid-cols-[92px_1fr] gap-2">
      <span className="shrink-0 text-white/40">{label}</span>
      <span
        className={`min-w-0 truncate ${mono ? "font-mono text-xs" : ""} ${
          ok === undefined ? "text-white/70" : ok ? "text-emerald-300/80" : "text-amber-300/70"
        }`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

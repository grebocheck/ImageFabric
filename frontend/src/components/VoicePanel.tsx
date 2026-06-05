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
const latencyPresets = [
  { id: "fast", label: "Fast", chunk: 96, crossFade: 0.03, extra: 3 },
  { id: "balanced", label: "Balanced", chunk: 133, crossFade: 0.05, extra: 5 },
  { id: "quality", label: "Quality", chunk: 192, crossFade: 0.08, extra: 7 },
];
const waveformSlots = 64;
const timingLabels = ["prep", "f0", "infer", "post", "io", "mix"];

type MeterSample = {
  input: number;
  output: number;
};

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

function meter(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "...";
  return `${Number(value).toFixed(1)} ms`;
}

function focusIsTextEntry(): boolean {
  const el = document.activeElement;
  if (!(el instanceof HTMLElement)) return false;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable;
}

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [modelId, setModelId] = useState("");
  const [pitch, setPitch] = useState(0);
  const [formantShift, setFormantShift] = useState(0);
  const [indexRatio, setIndexRatio] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Detector, setF0Detector] = useState("rmvpe_onnx");
  const [passThrough, setPassThrough] = useState(false);
  const [ptt, setPtt] = useState(false);
  const [inputDeviceId, setInputDeviceId] = useState(-1);
  const [outputDeviceId, setOutputDeviceId] = useState(-1);
  const [monitorDeviceId, setMonitorDeviceId] = useState(-1);
  const [sampleRate, setSampleRate] = useState(48000);
  const [readChunkSize, setReadChunkSize] = useState(133);
  const [crossFadeOverlap, setCrossFadeOverlap] = useState(0.05);
  const [extraConvert, setExtraConvert] = useState(5);
  const [inputGain, setInputGain] = useState(1);
  const [outputGain, setOutputGain] = useState(1);
  const [monitorGain, setMonitorGain] = useState(1);
  const [meterHistory, setMeterHistory] = useState<MeterSample[]>([]);

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

  const models = status?.models ?? [];
  const inputDevices = status?.audio_devices.inputs ?? [];
  const outputDevices = status?.audio_devices.outputs ?? [];
  const selected = useMemo(() => models.find((m) => m.id === modelId), [models, modelId]);
  const canReach = Boolean(status?.server_reachable);
  const canControl = canReach && models.length > 0 && !busy;
  const live = Boolean(status?.server_audio_enabled || status?.voice_lane_active);
  const streamStarted = Boolean(status?.server_audio_started);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (!status?.server_reachable || !live) return;
    const id = window.setInterval(() => {
      void refresh();
    }, 750);
    return () => window.clearInterval(id);
  }, [live, refresh, status?.server_reachable]);

  useEffect(() => {
    if (!status) return;
    setModelId((prev) => selectedModelId(status) || prev);
    setPitch(num(status.settings.tran, 0));
    setFormantShift(num(status.settings.formantShift, 0));
    setIndexRatio(num(status.settings.indexRatio, 1));
    setProtect(num(status.settings.protect, 0.5));
    const f0 = String(status.settings.f0Detector ?? "rmvpe_onnx");
    setF0Detector(f0Options.some((o) => o.value === f0) ? f0 : "rmvpe_onnx");
    setPassThrough(Boolean(status.settings.passThrough));
    setInputDeviceId(num(status.settings.serverInputDeviceId, -1));
    setOutputDeviceId(num(status.settings.serverOutputDeviceId, -1));
    setMonitorDeviceId(num(status.settings.serverMonitorDeviceId, -1));
    setSampleRate(num(status.settings.serverAudioSampleRate, 48000));
    setReadChunkSize(num(status.settings.serverReadChunkSize, 133));
    setCrossFadeOverlap(num(status.settings.crossFadeOverlapSize, 0.05));
    setExtraConvert(num(status.settings.extraConvertSize, 5));
    setInputGain(num(status.settings.serverInputAudioGain, 1));
    setOutputGain(num(status.settings.serverOutputAudioGain, 1));
    setMonitorGain(num(status.settings.serverMonitorAudioGain, 1));
  }, [status]);

  useEffect(() => {
    if (!status) return;
    const sample = {
      input: Math.max(0, Math.min(1, num(status.metrics.input_vu, 0))),
      output: Math.max(0, Math.min(1, num(status.metrics.output_vu, 0))),
    };
    setMeterHistory((prev) => [...prev.slice(-(waveformSlots - 1)), sample]);
  }, [status]);

  const body = (): VoiceSettingsUpdate => ({
    model_id: modelId || null,
    pitch,
    formant_shift: formantShift,
    index_ratio: indexRatio,
    protect,
    f0_detector: f0Detector,
    pass_through: passThrough,
    server_input_device_id: inputDeviceId,
    server_output_device_id: outputDeviceId,
    server_monitor_device_id: monitorDeviceId,
    server_audio_sample_rate: sampleRate,
    server_read_chunk_size: readChunkSize,
    cross_fade_overlap_size: crossFadeOverlap,
    extra_convert_size: extraConvert,
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

  const applyPatch = (label: string, patch: VoiceSettingsUpdate) => run(label, () => api.voiceApplySettings({ ...body(), ...patch }));

  const onLive = (next: boolean) => run(next ? "live-on" : "live-off", () => (
    next ? api.voiceStartSession(body()) : api.voiceStopSession()
  ));

  const onBypass = (next: boolean) => {
    setPassThrough(next);
    if (canReach) void applyPatch("bypass", { pass_through: next });
  };

  const onPtt = (next: boolean) => {
    setPtt(next);
    if (!canReach) return;
    if (next) {
      setPassThrough(true);
      void applyPatch("ptt", { pass_through: true });
    } else {
      void applyPatch("ptt", { pass_through: passThrough });
    }
  };

  const onPreset = (preset: (typeof latencyPresets)[number]) => {
    setReadChunkSize(preset.chunk);
    setCrossFadeOverlap(preset.crossFade);
    setExtraConvert(preset.extra);
    if (canReach) {
      void applyPatch("preset", {
        server_read_chunk_size: preset.chunk,
        cross_fade_overlap_size: preset.crossFade,
        extra_convert_size: preset.extra,
      });
    }
  };

  useEffect(() => {
    if (!ptt || !canReach) return;
    const onDown = (event: KeyboardEvent) => {
      if (event.code !== "Space" || focusIsTextEntry() || event.repeat) return;
      event.preventDefault();
      void api.voiceApplySettings({ pass_through: false }).then(setStatus).catch(() => {});
    };
    const onUp = (event: KeyboardEvent) => {
      if (event.code !== "Space" || focusIsTextEntry()) return;
      event.preventDefault();
      void api.voiceApplySettings({ pass_through: true }).then(setStatus).catch(() => {});
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, [canReach, ptt]);

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col gap-4 overflow-y-auto p-1">
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

        <div className="mb-4 grid gap-3 md:grid-cols-3">
          <Meter label="Input" value={meter(status?.metrics.input_vu ?? 0)} />
          <Meter label="Output" value={meter(status?.metrics.output_vu ?? 0)} />
          <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
            <div className="flex items-center justify-between text-xs">
              <span className="uppercase tracking-wide text-white/40">Latency</span>
              <span className="font-mono text-white/65">
                {formatMs(status?.metrics.total_ms ?? status?.metrics.chunk_ms)}
              </span>
            </div>
            <div className="mt-2 h-1.5 rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-sky-400/80"
                style={{ width: `${Math.min(100, Math.max(6, Number(status?.metrics.total_ms ?? status?.metrics.chunk_ms ?? 0)))}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mb-4 grid gap-3 lg:grid-cols-[1.2fr_0.8fr]">
          <WaveformMonitor samples={meterHistory} />
          <PerformanceBreakdown metrics={status?.metrics} />
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
            <div className="text-xs uppercase tracking-wide text-white/40">Formant</div>
            <Slider value={formantShift} min={-2} max={2} step={0.01} onChange={setFormantShift} />
          </div>

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

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-white/55">
              <Toggle checked={passThrough} onChange={onBypass} disabled={!canReach || Boolean(busy)} />
              Bypass
            </label>
            <label className="flex items-center gap-2 text-xs text-white/55">
              <Toggle checked={ptt} onChange={onPtt} disabled={!canReach} />
              PTT
            </label>
          </div>
          <div className="flex gap-1.5">
            {latencyPresets.map((preset) => (
              <button
                key={preset.id}
                onClick={() => onPreset(preset)}
                disabled={!canReach || Boolean(busy)}
                className="rounded border border-white/10 px-2 py-1 text-xs text-white/65 transition hover:bg-white/10 disabled:opacity-30"
              >
                {preset.label}
              </button>
            ))}
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

        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Crossfade</div>
            <Slider value={crossFadeOverlap} min={0} max={0.2} step={0.01} onChange={setCrossFadeOverlap} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Extra buffer</div>
            <Slider value={extraConvert} min={0} max={10} step={0.1} onChange={setExtraConvert} />
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

function WaveformMonitor({ samples }: { samples: MeterSample[] }) {
  const bars = Array.from({ length: waveformSlots }, (_, index) => {
    const offset = samples.length - waveformSlots + index;
    return offset >= 0 ? samples[offset] : { input: 0, output: 0 };
  });
  const latest = samples[samples.length - 1] ?? { input: 0, output: 0 };

  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="uppercase tracking-wide text-white/40">Waveform</span>
        <span className="font-mono text-white/55">
          in {meter(latest.input)}% / out {meter(latest.output)}%
        </span>
      </div>
      <div className="mt-3 flex h-24 items-center gap-px overflow-hidden rounded bg-black/25 px-2 py-2">
        {bars.map((sample, index) => {
          const inputHeight = sample.input > 0 ? Math.max(2, sample.input * 46) : 0;
          const outputHeight = sample.output > 0 ? Math.max(2, sample.output * 46) : 0;
          return (
            <div key={index} className="relative h-full min-w-0 flex-1">
              <div className="absolute left-0 right-0 top-1/2 h-px bg-white/10" />
              <div
                className="absolute bottom-1/2 left-0 right-0 rounded-t-sm bg-emerald-400/80"
                style={{ height: `${inputHeight}%` }}
              />
              <div
                className="absolute left-0 right-0 top-1/2 rounded-b-sm bg-sky-400/75"
                style={{ height: `${outputHeight}%` }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PerformanceBreakdown({ metrics }: { metrics?: VoiceStatus["metrics"] }) {
  const timings = (metrics?.timings_ms ?? []).filter((value) => Number.isFinite(value)).slice(0, 6);
  const total = metrics?.total_ms ?? null;
  const chunk = metrics?.chunk_ms ?? null;
  const max = Math.max(1, Number(total ?? 0), Number(chunk ?? 0), ...timings);

  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="uppercase tracking-wide text-white/40">Timing</span>
        <span className="font-mono text-white/55">{formatMs(total ?? chunk)}</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <MetricPill label="chunk" value={formatMs(chunk)} />
        <MetricPill label="total" value={formatMs(total)} />
      </div>
      <div className="mt-3 flex flex-col gap-2">
        {timings.length === 0 ? (
          <div className="rounded border border-white/10 bg-white/[0.03] px-2 py-1.5 text-xs text-white/35">waiting for stages</div>
        ) : (
          timings.map((value, index) => (
            <div key={`${index}-${value}`} className="min-w-0">
              <div className="flex items-center justify-between gap-3 text-[11px]">
                <span className="truncate uppercase tracking-wide text-white/35">{timingLabels[index] ?? `stage ${index + 1}`}</span>
                <span className="shrink-0 font-mono text-white/55">{formatMs(value)}</span>
              </div>
              <div className="mt-1 h-1 rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-violet-400/75"
                  style={{ width: `${Math.min(100, Math.max(4, (value / max) * 100))}%` }}
                />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded border border-white/10 bg-white/[0.03] px-2 py-1.5">
      <div className="truncate text-[10px] uppercase tracking-wide text-white/30">{label}</div>
      <div className="truncate font-mono text-xs text-white/65">{value}</div>
    </div>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between text-xs">
        <span className="uppercase tracking-wide text-white/40">{label}</span>
        <span className="font-mono text-white/65">{value}%</span>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-white/10">
        <div className="h-full rounded-full bg-emerald-400/80 transition-[width]" style={{ width: `${value}%` }} />
      </div>
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

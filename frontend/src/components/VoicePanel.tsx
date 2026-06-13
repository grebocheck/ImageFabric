import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import { Meter, WaveformMonitor, type MeterSample } from "./VoiceMeters";
import {
  DeviceSelect,
  LatencyMeter,
  MonitorSelect,
  OfflineDevice,
  RoutingApplyHint,
  Row,
  VoiceSlotList,
  type RoutingApplyState,
} from "./VoicePanelParts";
import {
  clearVoicePreset,
  denoiseOptions,
  deviceName,
  f0Options,
  feminineVoicePreset,
  formatBytes,
  formatMs,
  inputHighpassOptions,
  latencyPresets,
  meter,
  nativeRoutingSettingsPatch,
  nativeSettingsToVoiceState,
  nativeTuningSettingsPatch,
  nativeVoicePresetSettingsPatch,
  num,
  recommendedVoicePreset,
  resolveMonitorDeviceId,
  sampleRates,
  selectedNativeModelId,
  smoothVoicePreset,
  waveformSlots,
} from "./voiceHelpers";
import type {
  VoiceEngineAsset,
  VoiceEngineConvertResult,
  VoiceEnginePreset,
  VoiceEngineRecordingResult,
  VoiceEngineSettingsUpdate,
  VoiceEngineStatus,
  VoiceModel,
} from "../types";

const field = "w-full rounded-md border border-white/10 bg-black/25 px-2.5 py-1.5 text-sm outline-none transition focus:border-accent";
const assetSearchHint = "content_vec_500.onnx + rmvpe.pt -> models/voice/pretrain";
const denoiseAssetHint = "dtln_model_1.onnx + dtln_model_2.onnx -> models/voice/pretrain/denoise";
const modelDirHint = "models/voice";

const nativeF0Detectors = new Set(["rmvpe", "fcpe", "crepe_tiny", "crepe_full"]);
const nativeF0Options = f0Options.map((option) => (
  nativeF0Detectors.has(option.value)
    ? option
    : { ...option, disabled: true, hint: "unavailable" }
));

type Profile =
  | typeof recommendedVoicePreset
  | typeof clearVoicePreset
  | typeof smoothVoicePreset
  | typeof feminineVoicePreset;

function focusIsTextEntry(): boolean {
  const el = document.activeElement;
  if (!(el instanceof HTMLElement)) return false;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable;
}

function routingKey(body: VoiceEngineSettingsUpdate): string {
  return JSON.stringify(body);
}

function parseApiError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  const match = raw.match(/^(\d{3})\s+([\s\S]*)$/);
  const status = match?.[1] ?? "";
  let detail = match?.[2] ?? raw;
  try {
    const parsed = JSON.parse(detail) as { detail?: unknown };
    if (parsed.detail) detail = String(parsed.detail);
  } catch {
    // Keep the plain response text.
  }
  if (status === "415") return `Unsupported audio file: ${detail}`;
  if (status === "503") return `Voice engine is not ready: ${detail}`;
  if (status === "413") return `Audio file is too large: ${detail}`;
  return detail || raw;
}

function assetTitle(asset: VoiceEngineAsset): string {
  if (!asset.found && asset.name === "denoise_dtln") return denoiseAssetHint;
  return asset.found ? (asset.path ?? asset.name) : assetSearchHint;
}

function timingsLine(timings: Record<string, number>): string {
  const parts = Object.entries(timings)
    .filter(([, value]) => Number.isFinite(value))
    .map(([key, value]) => `${key} ${formatMs(value)}`);
  return parts.join(" / ") || "timings unavailable";
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
}

function signed(value: number, digits = 0): string {
  const fixed = value.toFixed(digits);
  return value > 0 ? `+${fixed}` : fixed;
}

function Panel({
  title,
  eyebrow,
  aside,
  children,
  className = "",
}: {
  title: string;
  eyebrow?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-white/10 bg-surface p-4 shadow-panel ${className}`}>
      <div className="mb-3 flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          {eyebrow ? <div className="text-[11px] font-medium text-white/35">{eyebrow}</div> : null}
          <h3 className="truncate text-sm font-semibold text-white/85">{title}</h3>
        </div>
        {aside ? <div className="shrink-0">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}

function StatusTile({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  tone?: "neutral" | "good" | "warn" | "info";
}) {
  const color = {
    neutral: "border-white/10 bg-black/20",
    good: "border-emerald-300/25 bg-emerald-300/10",
    warn: "border-amber-300/25 bg-amber-300/10",
    info: "border-sky-300/25 bg-sky-300/10",
  }[tone];
  return (
    <div className={`min-w-0 rounded-md border px-3 py-2 ${color}`}>
      <div className="truncate text-[11px] text-white/40">{label}</div>
      <div className="mt-0.5 truncate text-sm font-medium text-white/80">{value}</div>
    </div>
  );
}

function Button({
  children,
  onClick,
  disabled,
  tone = "ghost",
  className = "",
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "ghost" | "primary" | "danger" | "warn" | "success";
  className?: string;
  type?: "button" | "submit";
  title?: string;
}) {
  const tones = {
    ghost: "border-white/12 text-white/70 hover:bg-white/10 hover:text-white",
    primary: "border-accent/40 bg-accent text-white hover:bg-accent-hover",
    danger: "border-red-400/40 bg-red-600/90 text-white hover:bg-red-500",
    warn: "border-amber-300/30 bg-amber-300/10 text-amber-100 hover:bg-amber-300/15",
    success: "border-emerald-300/35 bg-emerald-600 text-white hover:bg-emerald-500",
  }[tone];
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-35 ${tones} ${className}`}
    >
      {children}
    </button>
  );
}

function MiniButton({
  children,
  onClick,
  active = false,
  disabled = false,
}: {
  children: ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded border px-2 py-1 text-xs transition disabled:opacity-30 ${
        active
          ? "border-accent/45 bg-accent/20 text-white"
          : "border-white/10 bg-black/15 text-white/62 hover:bg-white/10 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

function SignedControl({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = "",
  precision = 0,
  quick = [],
  disabled = false,
  note,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  unit?: string;
  precision?: number;
  quick?: number[];
  disabled?: boolean;
  note?: ReactNode;
}) {
  const commit = (next: number) => onChange(clamp(Number(next.toFixed(precision || 3)), min, max));
  return (
    <div className={`rounded-md border border-white/10 bg-black/15 p-3 ${disabled ? "opacity-55" : ""}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 text-xs font-medium text-white/55">{label}</div>
        <div className="shrink-0 font-mono text-lg font-semibold tabular-nums text-white/90">
          {signed(value, precision)}{unit}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-[34px_1fr_34px] items-center gap-2">
        <button
          type="button"
          onClick={() => commit(value - step)}
          disabled={disabled || value <= min}
          className="grid h-8 w-8 place-items-center rounded-md border border-white/10 bg-white/[0.03] text-lg leading-none text-white/70 transition hover:bg-white/10 disabled:opacity-25"
        >
          -
        </button>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(event) => commit(Number(event.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/15 accent-accent disabled:cursor-not-allowed"
        />
        <button
          type="button"
          onClick={() => commit(value + step)}
          disabled={disabled || value >= max}
          className="grid h-8 w-8 place-items-center rounded-md border border-white/10 bg-white/[0.03] text-lg leading-none text-white/70 transition hover:bg-white/10 disabled:opacity-25"
        >
          +
        </button>
      </div>
      {quick.length ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {quick.map((item) => (
            <MiniButton key={item} active={Math.abs(item - value) < step / 2} disabled={disabled} onClick={() => commit(item)}>
              {signed(item, precision)}{unit}
            </MiniButton>
          ))}
        </div>
      ) : null}
      {note ? <div className="mt-2 text-xs text-white/38">{note}</div> : null}
    </div>
  );
}

function CompactSignedControl({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  precision = 0,
  unit = "",
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  precision?: number;
  unit?: string;
}) {
  const commit = (next: number) => onChange(clamp(Number(next.toFixed(precision || 3)), min, max));
  return (
    <div className="min-w-0 rounded-md border border-white/10 bg-black/15 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium text-white/55">{label}</span>
        <span className="shrink-0 font-mono text-sm font-semibold tabular-nums text-white/85">
          {signed(value, precision)}{unit}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-[24px_1fr_24px] items-center gap-2">
        <button
          type="button"
          onClick={() => commit(value - step)}
          className="grid h-6 w-6 place-items-center rounded border border-white/10 text-sm text-white/65 hover:bg-white/10"
        >
          -
        </button>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => commit(Number(event.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/15 accent-accent"
        />
        <button
          type="button"
          onClick={() => commit(value + step)}
          className="grid h-6 w-6 place-items-center rounded border border-white/10 text-sm text-white/65 hover:bg-white/10"
        >
          +
        </button>
      </div>
    </div>
  );
}

function LabeledSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  valueLabel,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  valueLabel?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-center justify-between gap-2">
        <div className="truncate text-xs font-medium text-white/55">{label}</div>
        {valueLabel ? <div className="shrink-0 font-mono text-xs tabular-nums text-white/45">{valueLabel}</div> : null}
      </div>
      <Slider value={value} min={min} max={max} step={step} onChange={onChange} />
    </div>
  );
}

function presetMetric(preset: VoiceEnginePreset, key: keyof VoiceEngineSettingsUpdate, fallback = "..."): string {
  const value = preset.settings[key];
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") {
    if (key === "input_formant" || key === "index_ratio" || key === "noise_scale" || key === "f0_smoothing") {
      return value.toFixed(2);
    }
    return String(value);
  }
  return String(value);
}

function presetModelLabel(preset: VoiceEnginePreset, models: VoiceModel[]): string {
  if (!preset.model_id) return "settings only";
  return models.find((model) => model.id === preset.model_id)?.name ?? preset.model_id;
}

function PresetCard({
  preset,
  models,
  active,
  canApply,
  busy,
  onSelect,
  onApply,
  onUpdate,
  onDelete,
}: {
  preset: VoiceEnginePreset;
  models: VoiceModel[];
  active: boolean;
  canApply: boolean;
  busy: string;
  onSelect: () => void;
  onApply: () => void;
  onUpdate: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={`rounded-md border p-3 transition ${
        active ? "border-accent/45 bg-accent/10" : "border-white/10 bg-black/15 hover:bg-white/[0.04]"
      }`}
    >
      <button type="button" onClick={onSelect} className="block w-full min-w-0 text-left">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-white/86">{preset.name}</div>
            <div className="mt-1 truncate text-xs text-white/38" title={presetModelLabel(preset, models)}>
              {presetModelLabel(preset, models)}
            </div>
          </div>
          {preset.model_id ? <Badge color="bg-sky-700/45 text-sky-100">model</Badge> : <Badge>settings</Badge>}
        </div>
        <div className="mt-3 grid grid-cols-4 gap-1.5">
          <PresetMini label="pitch" value={presetMetric(preset, "pitch", "0")} />
          <PresetMini label="form" value={presetMetric(preset, "input_formant", "0.00")} />
          <PresetMini label="idx" value={presetMetric(preset, "index_ratio", "0.00")} />
          <PresetMini label="chunk" value={presetMetric(preset, "server_read_chunk_size", "...")} />
        </div>
      </button>
      <div className="mt-3 flex flex-wrap gap-1.5">
        <Button onClick={onApply} disabled={!canApply} tone={active ? "primary" : "ghost"} className="px-2 py-1 text-xs">
          {busy === "preset-apply" && active ? "Applying..." : "Apply"}
        </Button>
        <Button onClick={onUpdate} disabled={!canApply || !active} className="px-2 py-1 text-xs">
          {busy === "preset-update" && active ? "Updating..." : "Update"}
        </Button>
        <Button onClick={onDelete} disabled={!canApply || !active} tone="danger" className="px-2 py-1 text-xs">
          {busy === "preset-delete" && active ? "Deleting..." : "Delete"}
        </Button>
      </div>
    </div>
  );
}

function PresetMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded border border-white/10 bg-white/[0.03] px-2 py-1">
      <div className="truncate text-[10px] text-white/32">{label}</div>
      <div className="truncate font-mono text-[11px] tabular-nums text-white/68">{value}</div>
    </div>
  );
}

function DiagnosticsCompact({ status, samples }: { status: VoiceEngineStatus | null; samples: MeterSample[] }) {
  const metrics = status?.metrics;
  const timings = Object.entries(metrics?.timings_ms ?? {})
    .filter(([, value]) => Number.isFinite(value))
    .slice(0, 5);
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 gap-2">
        <StatusTile label="Total" value={formatMs(metrics?.total_ms ?? metrics?.chunk_ms)} />
        <StatusTile label="Chunk" value={formatMs(metrics?.chunk_ms)} />
        <StatusTile label="Overruns" value={metrics?.overruns ?? 0} tone={metrics?.overruns ? "warn" : "neutral"} />
        <StatusTile label="Squelch" value={metrics?.squelched ? "silence" : "voice"} tone={metrics?.squelched ? "warn" : "good"} />
      </div>
      <WaveformMonitor samples={samples} />
      <div className="grid gap-1.5">
        {timings.length ? timings.map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-3 rounded border border-white/10 bg-black/15 px-2 py-1 text-xs">
            <span className="truncate text-white/42">{label}</span>
            <span className="shrink-0 font-mono text-white/65">{formatMs(value)}</span>
          </div>
        )) : (
          <div className="rounded border border-white/10 bg-black/15 px-2 py-1.5 text-xs text-white/36">waiting for stages</div>
        )}
      </div>
    </div>
  );
}

function ModelBadges({ model }: { model: VoiceModel | null | undefined }) {
  if (!model) return <Badge>no voice</Badge>;
  return (
    <span className="flex flex-wrap gap-1.5">
      <Badge color="bg-accent/45 text-accent-fg">{model.type}{model.version ? ` ${model.version}` : ""}</Badge>
      <Badge color={model.f0 ? "bg-sky-700/50 text-sky-100" : "bg-white/10 text-white/55"}>
        {model.f0 ? "f0 pitch" : "no f0"}
      </Badge>
      <Badge color={model.has_index ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
        {model.has_index ? "index" : "no index"}
      </Badge>
      {model.sampling_rate ? <Badge>{model.sampling_rate} Hz</Badge> : null}
    </span>
  );
}

function VoiceOption({
  option,
  models,
}: {
  option: { value: string; label: string; hint?: string };
  models: VoiceModel[];
}) {
  const model = models.find((item) => item.id === option.value);
  return (
    <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
      <span className="min-w-0">
        <span className="block truncate">{option.label}</span>
        <span className="block truncate text-[11px] text-white/36">{model?.slot ?? option.hint}</span>
      </span>
      {model ? <ModelBadges model={model} /> : null}
    </span>
  );
}

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceEngineStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [modelId, setModelId] = useState("");
  const [pitch, setPitch] = useState(0);
  const [speakerId, setSpeakerId] = useState(0);
  const [formantShift, setFormantShift] = useState(0);
  const [inputGateDb, setInputGateDb] = useState(-90);
  const [inputHighpassHz, setInputHighpassHz] = useState(80);
  const [inputDenoise, setInputDenoise] = useState<"off" | "dtln">("off");
  const [silenceThresholdDb, setSilenceThresholdDb] = useState(-72);
  const [silenceHoldMs, setSilenceHoldMs] = useState(250);
  const [indexRatio, setIndexRatio] = useState(0.55);
  const [protect, setProtect] = useState(0.5);
  const [noiseScale, setNoiseScale] = useState(0.66666);
  const [f0Smoothing, setF0Smoothing] = useState(0);
  const [f0Detector, setF0Detector] = useState("fcpe");
  const [passThrough, setPassThrough] = useState(false);
  const [ptt, setPtt] = useState(false);
  const [inputDeviceId, setInputDeviceId] = useState(-1);
  const [outputDeviceId, setOutputDeviceId] = useState(-1);
  const [monitorDeviceId, setMonitorDeviceId] = useState(-1);
  const [sampleRate, setSampleRate] = useState(48000);
  const [readChunkSize, setReadChunkSize] = useState(133);
  const [crossFadeOverlap, setCrossFadeOverlap] = useState(0.05);
  const [extraConvert, setExtraConvert] = useState(2);
  const [inputGain, setInputGain] = useState(1);
  const [outputGain, setOutputGain] = useState(1);
  const [monitorGain, setMonitorGain] = useState(1);
  const [meterHistory, setMeterHistory] = useState<MeterSample[]>([]);
  const [voicesOpen, setVoicesOpen] = useState(false);
  const [tuningDirty, setTuningDirty] = useState(false);
  const [routingApplyState, setRoutingApplyState] = useState<RoutingApplyState>("idle");
  const [offlineFile, setOfflineFile] = useState<File | null>(null);
  const [offlineModelId, setOfflineModelId] = useState("");
  const [offlinePitch, setOfflinePitch] = useState(0);
  const [offlineFormant, setOfflineFormant] = useState(0);
  const [offlineBusy, setOfflineBusy] = useState(false);
  const [offlineError, setOfflineError] = useState("");
  const [offlineResult, setOfflineResult] = useState<VoiceEngineConvertResult | null>(null);
  const [recordingResult, setRecordingResult] = useState<VoiceEngineRecordingResult | null>(null);
  const [voicePresets, setVoicePresets] = useState<VoiceEnginePreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [presetName, setPresetName] = useState("");
  const lastAppliedRoutingKeyRef = useRef("");
  const routingApplySeq = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const next = await api.voiceEngineStatus();
      setStatus(next);
      setError("");
      return next;
    } catch (err) {
      setError(parseApiError(err) || "failed to load native voice status");
      return null;
    }
  }, []);

  const refreshPresets = useCallback(async () => {
    try {
      const next = await api.voiceEnginePresets();
      setVoicePresets(next);
      setSelectedPresetId((current) => (current && next.some((preset) => preset.id === current) ? current : next[0]?.id ?? ""));
    } catch {
      setVoicePresets([]);
    }
  }, []);

  const models = useMemo(() => status?.models ?? [], [status]);
  const inputDevices = status?.audio_devices.inputs ?? [];
  const outputDevices = status?.audio_devices.outputs ?? [];
  const selected = useMemo(() => models.find((m) => m.id === modelId), [models, modelId]);
  const loadedModel = useMemo(
    () => models.find((m) => m.id === status?.loaded_model),
    [models, status?.loaded_model],
  );
  const statusLoaded = Boolean(status);
  const ready = Boolean(status?.ready);
  const live = Boolean(status?.live);
  const monitorOn = monitorDeviceId >= 0;
  const recording = Boolean(status?.recording.active);
  const canApply = statusLoaded && !busy;
  const canGoLive = ready && Boolean(modelId) && !busy;
  const selectedPreset = useMemo(
    () => voicePresets.find((preset) => preset.id === selectedPresetId) ?? null,
    [selectedPresetId, voicePresets],
  );
  const sessionConfig = status?.session_config ?? null;
  const deviceMissing = status?.settings.device_missing ?? { input: false, output: false, monitor: false };
  const inputRestartPending = Boolean(
    live && sessionConfig && sessionConfig.server_input_device_id !== (inputDeviceId >= 0 ? inputDeviceId : null),
  );
  const outputRestartPending = Boolean(
    live && sessionConfig && sessionConfig.server_output_device_id !== (outputDeviceId >= 0 ? outputDeviceId : null),
  );
  const monitorRestartPending = Boolean(
    live
      && sessionConfig
      && (sessionConfig.server_monitor_device_id == null || sessionConfig.server_monitor_device_id < 0
        ? -1
        : sessionConfig.server_monitor_device_id) !== (monitorDeviceId >= 0 ? monitorDeviceId : -1),
  );
  const sampleRateRestartPending = Boolean(
    live && sessionConfig && sessionConfig.server_audio_sample_rate !== sampleRate,
  );
  const chunkRestartPending = Boolean(
    live && sessionConfig && sessionConfig.server_read_chunk_size !== readChunkSize,
  );

  const routingPatch = useMemo(() => nativeRoutingSettingsPatch({
    inputDeviceId,
    outputDeviceId,
    monitorDeviceId,
    sampleRate,
    readChunkSize,
    crossFadeOverlap,
    extraConvert,
    inputGain,
    outputGain,
    monitorGain,
  }), [
    crossFadeOverlap,
    extraConvert,
    inputDeviceId,
    inputGain,
    monitorDeviceId,
    monitorGain,
    outputDeviceId,
    outputGain,
    readChunkSize,
    sampleRate,
  ]);
  const currentRoutingKey = useMemo(() => routingKey(routingPatch), [routingPatch]);

  useEffect(() => {
    void refresh();
    void refreshPresets();
  }, [refresh, refreshPresets]);

  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => {
      void refresh();
    }, 750);
    return () => window.clearInterval(id);
  }, [live, refresh]);

  useEffect(() => {
    if (!status) return;
    const nextModelId = selectedNativeModelId(status.models, modelId, status.loaded_model);
    setModelId(nextModelId);
    setOfflineModelId((prev) => selectedNativeModelId(status.models, prev || nextModelId, status.loaded_model) || prev);
    const next = nativeSettingsToVoiceState(status.settings);

    if (!tuningDirty) {
      setPitch(next.pitch);
      setOfflinePitch(next.pitch);
      setSpeakerId(next.speakerId);
      setFormantShift(next.formantShift);
      setOfflineFormant(next.formantShift);
      setInputGateDb(next.inputGateDb);
      setInputHighpassHz(next.inputHighpassHz);
      setInputDenoise(next.inputDenoise);
      setSilenceThresholdDb(next.silenceThresholdDb);
      setSilenceHoldMs(next.silenceHoldMs);
      setIndexRatio(next.indexRatio);
      setProtect(next.protect);
      setNoiseScale(next.noiseScale);
      setF0Smoothing(next.f0Smoothing);
      setF0Detector(next.f0Detector);
      setPassThrough(next.passThrough);
    }

    if (routingApplyState !== "pending" && routingApplyState !== "applying") {
      setInputDeviceId(next.inputDeviceId);
      setOutputDeviceId(next.outputDeviceId);
      setMonitorDeviceId(next.monitorDeviceId);
      setSampleRate(next.sampleRate);
      setReadChunkSize(next.readChunkSize);
      setCrossFadeOverlap(next.crossFadeOverlap);
      setExtraConvert(next.extraConvert);
      setInputGain(next.inputGain);
      setOutputGain(next.outputGain);
      setMonitorGain(next.monitorGain);
      lastAppliedRoutingKeyRef.current = routingKey(nativeRoutingSettingsPatch(next));
    }
  }, [modelId, routingApplyState, status, tuningDirty]);

  useEffect(() => {
    if (!status) return;
    const sample = {
      input: Math.max(0, Math.min(1, num(status.metrics.input_vu, 0))),
      output: Math.max(0, Math.min(1, num(status.metrics.output_vu, 0))),
    };
    setMeterHistory((prev) => [...prev.slice(-(waveformSlots - 1)), sample]);
  }, [status]);

  useEffect(() => {
    if (!statusLoaded) {
      setRoutingApplyState("idle");
      return;
    }
    if (!lastAppliedRoutingKeyRef.current) {
      lastAppliedRoutingKeyRef.current = currentRoutingKey;
      return;
    }
    if (currentRoutingKey === lastAppliedRoutingKeyRef.current) return;

    setRoutingApplyState("pending");
    const requestKey = currentRoutingKey;
    const seq = routingApplySeq.current + 1;
    routingApplySeq.current = seq;
    const id = window.setTimeout(async () => {
      if (requestKey === lastAppliedRoutingKeyRef.current) {
        setRoutingApplyState("applied");
        return;
      }
      setRoutingApplyState("applying");
      setError("");
      try {
        const next = await api.voiceEngineSettings(routingPatch);
        if (seq !== routingApplySeq.current) return;
        lastAppliedRoutingKeyRef.current = requestKey;
        setStatus(next);
        setRoutingApplyState("applied");
      } catch (err) {
        if (seq !== routingApplySeq.current) return;
        setRoutingApplyState("error");
        setError(parseApiError(err));
      }
    }, 400);
    return () => window.clearTimeout(id);
  }, [currentRoutingKey, routingPatch, statusLoaded]);

  const markTuning = () => setTuningDirty(true);
  const setDraftPitch = (value: number) => {
    const next = clamp(Math.round(value), -24, 24);
    setPitch(next);
    setOfflinePitch(next);
    markTuning();
  };
  const setDraftFormant = (value: number) => {
    const next = clamp(value, -2, 2);
    setFormantShift(next);
    setOfflineFormant(next);
    markTuning();
  };
  const setDraftSpeakerId = (value: number) => {
    setSpeakerId(clamp(Math.round(value), 0, 255));
    markTuning();
  };

  const tuningPatch = (): VoiceEngineSettingsUpdate => nativeTuningSettingsPatch({
    pitch,
    speakerId,
    formantShift,
    inputGateDb,
    inputHighpassHz,
    inputDenoise,
    silenceThresholdDb,
    silenceHoldMs,
    indexRatio,
    protect,
    noiseScale,
    f0Smoothing,
    f0Detector,
    passThrough,
  });

  const fullSettingsPatch = (): VoiceEngineSettingsUpdate => ({
    ...tuningPatch(),
    ...routingPatch,
  });

  const presetSettingsPatch = (): VoiceEngineSettingsUpdate => nativeVoicePresetSettingsPatch({
    pitch,
    speakerId,
    formantShift,
    inputGateDb,
    inputHighpassHz,
    inputDenoise,
    silenceThresholdDb,
    silenceHoldMs,
    indexRatio,
    protect,
    noiseScale,
    f0Smoothing,
    f0Detector,
    sampleRate,
    readChunkSize,
    crossFadeOverlap,
    extraConvert,
    inputGain,
    outputGain,
    monitorGain,
  });

  async function run(label: string, fn: () => Promise<VoiceEngineStatus | null | void>) {
    setBusy(label);
    setError("");
    try {
      const next = await fn();
      if (next) setStatus(next);
    } catch (err) {
      setError(parseApiError(err));
    } finally {
      setBusy("");
    }
  }

  const onApply = () => run("apply", async () => {
    const next = await api.voiceEngineSettings(fullSettingsPatch());
    setTuningDirty(false);
    return next;
  });

  const applyPatch = (label: string, patch: VoiceEngineSettingsUpdate, syncTuning = false) => run(label, async () => {
    const next = await api.voiceEngineSettings(patch);
    if (syncTuning) setTuningDirty(false);
    return next;
  });

  const onLive = (next: boolean) => run(next ? "live-on" : "live-off", async () => {
    if (!next) return api.voiceEngineSessionStop();
    if (!modelId) throw new Error("Select a voice model before starting live mode");
    await api.voiceEngineSettings(fullSettingsPatch());
    setTuningDirty(false);
    return api.voiceEngineSessionStart(modelId);
  });

  const onRestartLive = () => run("live-restart", async () => {
    const nextModelId = modelId || status?.loaded_model;
    if (!nextModelId) throw new Error("Select a voice model before restarting live mode");
    await api.voiceEngineSessionStop();
    await api.voiceEngineSettings(fullSettingsPatch());
    setTuningDirty(false);
    return api.voiceEngineSessionStart(nextModelId);
  });

  const onMonitor = (next: boolean) => {
    if (!next) {
      setMonitorDeviceId(-1);
      return;
    }
    const resolved = resolveMonitorDeviceId(monitorDeviceId, outputDeviceId, outputDevices);
    if (resolved < 0) {
      setError("No output device is available for monitoring");
      return;
    }
    setMonitorDeviceId(resolved);
  };

  const onBypass = (next: boolean) => {
    setPassThrough(next);
    markTuning();
    if (statusLoaded) void applyPatch("bypass", { pass_through: next }, true);
  };

  const onPtt = (next: boolean) => {
    setPtt(next);
    if (!statusLoaded) return;
    if (next) {
      setPassThrough(true);
      void applyPatch("ptt", { pass_through: true }, true);
    } else {
      void applyPatch("ptt", { pass_through: passThrough }, true);
    }
  };

  const onPreset = (preset: (typeof latencyPresets)[number]) => {
    setReadChunkSize(preset.chunk);
    setCrossFadeOverlap(preset.crossFade);
    setExtraConvert(preset.extra);
  };

  const applyQualityProfile = (label: string, profile: Profile, pitchOverride?: number) => {
    const nextPitch = pitchOverride ?? pitch;
    setPitch(nextPitch);
    setOfflinePitch(nextPitch);
    setFormantShift(profile.formantShift);
    setOfflineFormant(profile.formantShift);
    setInputDenoise(profile.inputDenoise);
    setInputHighpassHz(profile.inputHighpassHz);
    setInputGateDb(profile.inputGateDb);
    setSilenceThresholdDb(profile.silenceThresholdDb);
    setSilenceHoldMs(profile.silenceHoldMs);
    setIndexRatio(profile.indexRatio);
    setProtect(profile.protect);
    setNoiseScale(profile.noiseScale);
    setF0Smoothing(profile.f0Smoothing);
    setReadChunkSize(profile.readChunkSize);
    setCrossFadeOverlap(profile.crossFadeOverlap);
    setExtraConvert(profile.extraConvert);
    setSampleRate(profile.sampleRate);
    void applyPatch(label, {
      pitch: nextPitch,
      input_formant: profile.formantShift,
      input_denoise: profile.inputDenoise,
      input_highpass_hz: profile.inputHighpassHz,
      input_gate_db: profile.inputGateDb,
      silence_threshold_db: profile.silenceThresholdDb,
      silence_hold_ms: profile.silenceHoldMs,
      index_ratio: profile.indexRatio,
      protect: profile.protect,
      noise_scale: profile.noiseScale,
      f0_smoothing: profile.f0Smoothing,
      server_read_chunk_size: profile.readChunkSize,
      cross_fade_overlap_size: profile.crossFadeOverlap,
      extra_convert_size: profile.extraConvert,
      server_audio_sample_rate: profile.sampleRate,
    }, true);
  };

  const onRecommended = () => applyQualityProfile("recommended", recommendedVoicePreset);
  const onClear = () => applyQualityProfile("clear-preset", clearVoicePreset);
  const onSmooth = () => applyQualityProfile("smooth-preset", smoothVoicePreset);
  const onFeminine = () => applyQualityProfile("female-preset", feminineVoicePreset, feminineVoicePreset.pitch);

  const selectVoicePreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    const preset = voicePresets.find((item) => item.id === presetId);
    if (preset) setPresetName(preset.name);
  };

  const onSaveVoicePreset = () => run("preset-save", async () => {
    const saved = await api.voiceEnginePresetCreate({
      name: presetName,
      model_id: modelId || null,
      settings: presetSettingsPatch(),
    });
    const next = await api.voiceEnginePresets();
    setVoicePresets(next);
    setSelectedPresetId(saved.id);
    setPresetName(saved.name);
    return null;
  });

  const applyVoicePreset = (preset: VoiceEnginePreset) => run("preset-apply", async () => {
    setSelectedPresetId(preset.id);
    setPresetName(preset.name);
    if (preset.model_id && models.some((model) => model.id === preset.model_id)) {
      setModelId(preset.model_id);
      setOfflineModelId(preset.model_id);
    }
    const next = await api.voiceEngineSettings(preset.settings);
    setTuningDirty(false);
    return next;
  });

  const onUpdateVoicePreset = () => run("preset-update", async () => {
    if (!selectedPreset) throw new Error("Choose a saved preset first");
    const updated = await api.voiceEnginePresetUpdate(selectedPreset.id, {
      name: presetName.trim() || selectedPreset.name,
      model_id: modelId || null,
      settings: presetSettingsPatch(),
    });
    const nextPresets = await api.voiceEnginePresets();
    setVoicePresets(nextPresets);
    setSelectedPresetId(updated.id);
    setPresetName(updated.name);
    setTuningDirty(false);
    return null;
  });

  const onDeleteVoicePreset = () => run("preset-delete", async () => {
    if (!selectedPreset) throw new Error("Choose a saved preset first");
    await api.voiceEnginePresetDelete(selectedPreset.id);
    await refreshPresets();
    setPresetName("");
    return null;
  });

  const onRecording = (next: boolean) => run(next ? "record-on" : "record-off", async () => {
    if (next) {
      setRecordingResult(null);
      return api.voiceEngineRecordingStart();
    }
    const updated = await api.voiceEngineRecordingStop();
    setRecordingResult(updated.recording_result ?? null);
    return updated;
  });

  const onOfflineConvert = async () => {
    if (!offlineFile) {
      setOfflineError("Choose a WAV, FLAC, OGG, or MP3 file first");
      return;
    }
    if (!offlineModelId) {
      setOfflineError("Choose a voice model first");
      return;
    }
    const form = new FormData();
    form.append("file", offlineFile);
    form.append("model_id", offlineModelId);
    form.append("pitch", String(offlinePitch));
    form.append("speaker_id", String(speakerId));
    form.append("index_ratio", String(indexRatio));
    form.append("protect", String(protect));
    form.append("noise_scale", String(noiseScale));
    form.append("f0_smoothing", String(f0Smoothing));
    form.append("input_highpass_hz", String(inputHighpassHz));
    form.append("input_gate_db", String(inputGateDb));
    form.append("input_formant", String(offlineFormant));
    form.append("input_denoise", inputDenoise);
    setOfflineBusy(true);
    setOfflineError("");
    setOfflineResult(null);
    try {
      setOfflineResult(await api.voiceEngineConvert(form));
    } catch (err) {
      setOfflineError(parseApiError(err));
    } finally {
      setOfflineBusy(false);
    }
  };

  useEffect(() => {
    if (!ptt || !statusLoaded) return;
    const onDown = (event: KeyboardEvent) => {
      if (event.code !== "Space" || focusIsTextEntry() || event.repeat) return;
      event.preventDefault();
      void api.voiceEngineSettings({ pass_through: false }).then(setStatus).catch(() => {});
    };
    const onUp = (event: KeyboardEvent) => {
      if (event.code !== "Space" || focusIsTextEntry()) return;
      event.preventDefault();
      void api.voiceEngineSettings({ pass_through: true }).then(setStatus).catch(() => {});
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, [ptt, statusLoaded]);

  const assetsFound = (status?.assets ?? []).filter((asset) => asset.found || asset.optional).length;
  const totalAssets = status?.assets.length ?? 0;
  const voiceOptions = models.map((m) => ({ value: m.id, label: m.name, hint: `${m.source ?? "local"} ${m.slot}` }));
  const selectedSupportsPitch = selected?.f0 !== false;

  return (
    <div className="flex h-full w-full flex-col gap-4 overflow-y-auto p-1">
      <header className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto]">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-xl font-semibold text-white/90">Voice Changer</h2>
            <Badge color={live ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
              {live ? "live" : "idle"}
            </Badge>
            {tuningDirty ? <Badge color="bg-amber-600/40 text-amber-100">unsaved tuning</Badge> : null}
          </div>
          <p className="mt-1 truncate text-sm text-white/45">
            {selected?.name ?? "No voice selected"} {selected ? "->" : ""} {live ? "microphone lane active" : "ready for setup"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => void refresh()} disabled={Boolean(busy)}>Refresh</Button>
          <Button onClick={onApply} disabled={!canApply} tone={tuningDirty ? "primary" : "ghost"}>
            {busy === "apply" ? "Applying..." : tuningDirty ? "Apply Changes" : "Apply"}
          </Button>
        </div>
      </header>

      {error ? (
        <div className="rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">{error}</div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(320px,0.92fr)_minmax(0,1.45fr)]">
        <Panel
          title="Voice And Engine"
          aside={(
            <Badge color={ready ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
              {ready ? "ready" : "missing"}
            </Badge>
          )}
        >
          <div className="grid gap-2 sm:grid-cols-2">
            <StatusTile label="Engine" value={status?.engine ?? "native-rvc"} tone={ready ? "good" : "warn"} />
            <StatusTile label="Mode" value={status?.stub ? "stub" : "real"} tone={status?.stub ? "info" : "neutral"} />
            <StatusTile label="Device" value={status?.device ?? "..."} />
            <StatusTile label="Assets" value={`${assetsFound}/${totalAssets || "..."}`} tone={ready ? "good" : "warn"} />
          </div>

          <div className="mt-4">
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <div className="text-xs font-medium text-white/55">Voice model</div>
              <Badge>{models.length} slots</Badge>
            </div>
            <Select
              value={modelId}
              onChange={(value) => {
                setModelId(value);
                setOfflineModelId(value);
              }}
              placeholder="no voices"
              options={voiceOptions}
              renderOption={(option) => <VoiceOption option={option} models={models} />}
            />
          </div>

          <div className="mt-3 rounded-md border border-white/10 bg-black/15 p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-white/82">{selected?.name ?? "No model selected"}</div>
                <div className="mt-1 truncate text-xs text-white/38">
                  loaded: {loadedModel?.name ?? status?.loaded_model ?? "none"}
                </div>
              </div>
              <ModelBadges model={selected} />
            </div>
            {selected ? (
              <div className="mt-3 grid gap-1.5 text-sm sm:grid-cols-2">
                <Row label="Slot" value={selected.slot} />
                <Row label="Size" value={formatBytes(selected.size_bytes)} />
                <Row label="Source" value={selected.source ?? "local"} />
                <Row label="Pitch" value={selected.f0 ? "active" : "model-limited"} ok={selected.f0} />
              </div>
            ) : null}
          </div>

          <button
            type="button"
            onClick={() => setVoicesOpen((open) => !open)}
            className="mt-3 w-full rounded-md border border-white/10 px-2.5 py-1.5 text-left text-xs font-medium text-white/55 transition hover:bg-white/10 hover:text-white/75"
          >
            {voicesOpen ? "Hide model list" : "Show model list"}
          </button>
          {voicesOpen ? <VoiceSlotList models={models} modelId={modelId} modelDir={modelDirHint} onSelect={setModelId} /> : null}

          <div className="mt-3 grid gap-1.5">
            {(status?.assets ?? []).map((asset) => (
              <div
                key={asset.name}
                title={assetTitle(asset)}
                className="flex items-center justify-between gap-2 rounded-md border border-white/10 bg-black/15 px-2.5 py-1.5"
              >
                <span className="min-w-0 truncate text-sm text-white/72">{asset.name}</span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <Badge color={asset.found ? "bg-emerald-700/55 text-emerald-100" : asset.optional ? "bg-white/10 text-white/55" : "bg-amber-600/40 text-amber-100"}>
                    {asset.found ? "found" : asset.optional ? "optional" : "missing"}
                  </Badge>
                  {asset.source ? <Badge>{asset.source}</Badge> : null}
                </span>
              </div>
            ))}
            {!status?.assets?.length ? <div className="text-sm text-white/40">Loading native assets...</div> : null}
          </div>
        </Panel>

        <Panel
          title="Live Console"
          aside={(
            <Badge color={live ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
              {live ? "on air" : "stopped"}
            </Badge>
          )}
        >
          {status?.session_error ? (
            <div className="mb-3 rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">
              {status.session_error}
            </div>
          ) : null}

          <div className={`rounded-md border p-4 ${live ? "border-emerald-400/30 bg-emerald-400/10" : "border-white/10 bg-black/15"}`}>
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto]">
              <div className="min-w-0">
                <div className="truncate text-base font-semibold text-white/88">
                  {live ? "Live voice is running" : "Live voice is off"}
                </div>
                <div className="mt-1 truncate text-sm text-white/42">
                  {deviceName(inputDevices, inputDeviceId, "input")}{" -> "}{selected?.name ?? "voice"}{" -> "}{deviceName(outputDevices, outputDeviceId, "output")}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {live ? (
                  <>
                    <Button onClick={() => onLive(false)} disabled={Boolean(busy) || recording} tone="danger">
                      {busy === "live-off" ? "Stopping..." : "Stop"}
                    </Button>
                    <Button onClick={onRestartLive} disabled={Boolean(busy) || recording} tone="warn">
                      {busy === "live-restart" ? "Restarting..." : "Restart"}
                    </Button>
                  </>
                ) : (
                  <Button onClick={() => onLive(true)} disabled={!canGoLive} tone="success">
                    {busy === "live-on" ? "Starting..." : "Start Live"}
                  </Button>
                )}
              </div>
            </div>
            {!live && !canGoLive ? (
              <div className="mt-2 text-xs text-amber-200/75">
                {busy ? "busy..." : !ready ? assetSearchHint : !modelId ? "select a voice model" : "cannot start right now"}
              </div>
            ) : null}
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <Meter label="Input" value={meter(status?.metrics.input_vu ?? 0)} />
            <Meter label={monitorOn ? "Output / Monitor" : "Output"} value={meter(status?.metrics.output_vu ?? 0)} tone="sky" />
            <LatencyMeter value={status?.metrics.total_ms ?? status?.metrics.chunk_ms} />
          </div>

          <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(250px,0.75fr)]">
            <div className={`rounded-md border px-3 py-2 ${recording ? "border-red-300/35 bg-red-400/10" : "border-white/10 bg-black/15"}`}>
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white/80">Recorder</span>
                    <Badge color={recording ? "bg-red-600/60 text-red-50" : "bg-white/10 text-white/55"}>
                      {recording ? `${(status?.recording.duration_s ?? 0).toFixed(1)} s` : "ready"}
                    </Badge>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-white/36">
                    {recordingResult ? `${recordingResult.sample_rate} Hz / ${recordingResult.duration_s.toFixed(2)} s` : "live output"}
                  </div>
                </div>
                <Button onClick={() => onRecording(!recording)} disabled={!live || Boolean(busy)} tone={recording ? "danger" : "ghost"}>
                  {busy === "record-on" ? "Starting..." : busy === "record-off" ? "Saving..." : recording ? "Save" : "Record"}
                </Button>
              </div>
              {recordingResult ? (
                <div className="mt-3">
                  <audio controls src={recordingResult.url} className="w-full" />
                  <div className="mt-2 flex justify-end gap-2">
                    <a href={recordingResult.url} download className="rounded border border-white/15 px-2 py-1 text-xs text-white/70 transition hover:bg-white/10 hover:text-white">WAV</a>
                    <a href={recordingResult.mp3_url} download className="rounded border border-white/15 px-2 py-1 text-xs text-white/70 transition hover:bg-white/10 hover:text-white">MP3</a>
                  </div>
                </div>
              ) : null}
            </div>

            <div className={`rounded-md border px-3 py-2 ${monitorOn ? "border-sky-400/30 bg-sky-400/10" : "border-white/10 bg-black/15"}`}>
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white/80">Monitor</span>
                    <Badge color={monitorOn ? "bg-sky-700/50 text-sky-100" : "bg-white/10 text-white/55"}>
                      {monitorOn ? "on" : "off"}
                    </Badge>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-white/36" title={deviceName(outputDevices, monitorDeviceId, "Off")}>
                    {deviceName(outputDevices, monitorDeviceId, "Off")}
                  </div>
                </div>
                <Toggle checked={monitorOn} onChange={onMonitor} disabled={!statusLoaded || outputDevices.length === 0} ariaLabel="Toggle monitor" />
              </div>
              <div className="mt-2">
                <LabeledSlider label="Monitor gain" value={monitorGain} min={0} max={2} step={0.01} onChange={setMonitorGain} />
              </div>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5">
            <Badge>overruns {status?.metrics.overruns ?? 0}</Badge>
            <Badge>underruns {status?.metrics.underruns ?? 0}</Badge>
            <Badge>chunk {formatMs(status?.metrics.chunk_ms)}</Badge>
            <Badge color={status?.metrics.squelched ? "bg-amber-600/40 text-amber-100" : "bg-emerald-700/45 text-emerald-100"}>
              {status?.metrics.squelched ? "silence" : "voice"}
            </Badge>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)]">
        <Panel
          title="Tuning"
          eyebrow={selected?.name ?? "no voice selected"}
          aside={(
            <div className="flex flex-wrap justify-end gap-1.5">
              <Button onClick={onRecommended} disabled={!canApply} tone="ghost" className="px-2 py-1 text-xs">
                {busy === "recommended" ? "Applying..." : "Baseline"}
              </Button>
              <Button onClick={onClear} disabled={!canApply} tone="ghost" className="px-2 py-1 text-xs">
                {busy === "clear-preset" ? "Applying..." : "Clear"}
              </Button>
              <Button onClick={onSmooth} disabled={!canApply} tone="ghost" className="px-2 py-1 text-xs">
                {busy === "smooth-preset" ? "Applying..." : "Smooth"}
              </Button>
              <Button onClick={onFeminine} disabled={!canApply} tone="ghost" className="px-2 py-1 text-xs">
                {busy === "female-preset" ? "Applying..." : "Female +12"}
              </Button>
            </div>
          )}
        >
          <div className="grid gap-3 lg:grid-cols-[minmax(260px,0.75fr)_minmax(0,1fr)]">
            <SignedControl
              label="Pitch"
              value={pitch}
              min={-24}
              max={24}
              step={1}
              onChange={setDraftPitch}
              unit=" st"
              quick={[-12, -7, 0, 7, 12]}
              note={selectedSupportsPitch ? "f0 model" : "no-f0 model"}
            />
            <SignedControl
              label="Formant"
              value={formantShift}
              min={-2}
              max={2}
              step={0.05}
              precision={2}
              onChange={setDraftFormant}
              quick={[-0.5, 0, 0.5]}
            />
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <label className="min-w-0">
              <div className="mb-1.5 text-xs font-medium text-white/55">F0 detector</div>
              <Select
                value={f0Detector}
                onChange={(value) => {
                  setF0Detector(value);
                  markTuning();
                }}
                options={nativeF0Options}
              />
            </label>
            <label className="min-w-0">
              <div className="mb-1.5 text-xs font-medium text-white/55">Speaker ID</div>
              <input
                type="number"
                min={0}
                max={255}
                step={1}
                value={speakerId}
                onChange={(event) => setDraftSpeakerId(Number(event.target.value))}
                className={field}
              />
            </label>
            <label className="min-w-0">
              <div className="mb-1.5 text-xs font-medium text-white/55">Denoise</div>
              <Select
                value={inputDenoise}
                onChange={(value) => {
                  setInputDenoise(value === "dtln" ? "dtln" : "off");
                  markTuning();
                }}
                options={denoiseOptions}
              />
            </label>
            <label className="min-w-0">
              <div className="mb-1.5 text-xs font-medium text-white/55">High-pass</div>
              <Select
                value={String(inputHighpassHz)}
                onChange={(value) => {
                  setInputHighpassHz(Number(value));
                  markTuning();
                }}
                options={inputHighpassOptions}
              />
            </label>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <LabeledSlider label="Index ratio" value={indexRatio} min={0} max={1} step={0.01} onChange={(value) => { setIndexRatio(value); markTuning(); }} valueLabel={indexRatio.toFixed(2)} />
            <LabeledSlider label="Protect" value={protect} min={0} max={1} step={0.01} onChange={(value) => { setProtect(value); markTuning(); }} valueLabel={protect.toFixed(2)} />
            <LabeledSlider label="Noise scale" value={noiseScale} min={0} max={1} step={0.01} onChange={(value) => { setNoiseScale(value); markTuning(); }} valueLabel={noiseScale.toFixed(2)} />
            <LabeledSlider label="F0 smooth" value={f0Smoothing} min={0} max={1} step={0.01} onChange={(value) => { setF0Smoothing(value); markTuning(); }} valueLabel={f0Smoothing.toFixed(2)} />
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <LabeledSlider
              label="Noise gate"
              value={inputGateDb}
              min={-90}
              max={-20}
              step={1}
              onChange={(value) => { setInputGateDb(value); markTuning(); }}
              valueLabel={inputGateDb <= -90 ? "off" : `${inputGateDb.toFixed(0)} dB`}
            />
            <LabeledSlider
              label="Idle squelch"
              value={silenceThresholdDb}
              min={-90}
              max={-20}
              step={1}
              onChange={(value) => { setSilenceThresholdDb(value); markTuning(); }}
              valueLabel={silenceThresholdDb <= -90 ? "off" : `${silenceThresholdDb.toFixed(0)} dB`}
            />
            <LabeledSlider
              label="Hold"
              value={silenceHoldMs}
              min={0}
              max={2000}
              step={50}
              onChange={(value) => { setSilenceHoldMs(value); markTuning(); }}
              valueLabel={`${Math.round(silenceHoldMs)} ms`}
            />
            <div className="flex items-end justify-between gap-3 rounded-md border border-white/10 bg-black/15 px-3 py-2">
              <label className="flex items-center gap-2 text-sm text-white/62">
                <Toggle checked={passThrough} onChange={onBypass} disabled={!statusLoaded || Boolean(busy)} />
                Bypass
              </label>
              <label className="flex items-center gap-2 text-sm text-white/62">
                <Toggle checked={ptt} onChange={onPtt} disabled={!statusLoaded} />
                PTT
              </label>
            </div>
          </div>
        </Panel>

        <Panel
          title="Routing And Timing"
          aside={<RoutingApplyHint canReach={statusLoaded} state={routingApplyState} />}
        >
          <div className="grid gap-3">
            {inputDevices.length ? (
              <DeviceSelect
                label="Input"
                value={inputDeviceId}
                devices={inputDevices}
                fallback="No input selected"
                missing={deviceMissing.input}
                restartPending={inputRestartPending}
                onChange={setInputDeviceId}
              />
            ) : (
              <OfflineDevice label="Input" message={statusLoaded ? "No input devices reported" : "Loading devices"} />
            )}
            {outputDevices.length ? (
              <DeviceSelect
                label="Output"
                value={outputDeviceId}
                devices={outputDevices}
                fallback="No output selected"
                missing={deviceMissing.output}
                restartPending={outputRestartPending}
                onChange={setOutputDeviceId}
              />
            ) : (
              <OfflineDevice label="Output" message={statusLoaded ? "No output devices reported" : "Loading devices"} />
            )}
            {outputDevices.length ? (
              <MonitorSelect
                value={monitorDeviceId}
                devices={outputDevices}
                missing={deviceMissing.monitor}
                restartPending={monitorRestartPending}
                onChange={setMonitorDeviceId}
              />
            ) : (
              <OfflineDevice label="Monitor" message={statusLoaded ? "No output device for monitor" : "Loading devices"} />
            )}
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <label>
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-white/55">Sample rate</span>
                {sampleRateRestartPending ? <span className="text-[11px] text-amber-200/70">restart</span> : null}
              </div>
              <Select
                value={String(sampleRate)}
                onChange={(v) => setSampleRate(Number(v))}
                options={sampleRates.map((rate) => ({ value: String(rate), label: `${rate} Hz` }))}
              />
            </label>

            <label>
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-white/55">Chunk</span>
                {chunkRestartPending ? <span className="text-[11px] text-amber-200/70">restart</span> : null}
              </div>
              <input
                type="number"
                min={1}
                max={1024}
                value={readChunkSize}
                onChange={(event) => setReadChunkSize(clamp(Number(event.target.value), 1, 1024))}
                className={field}
              />
            </label>
          </div>

          <div className="mt-4 grid gap-4">
            <LabeledSlider label="Extra buffer" value={extraConvert} min={0} max={10} step={0.1} onChange={setExtraConvert} valueLabel={`${extraConvert.toFixed(1)} s`} />
            <LabeledSlider label="Crossfade" value={crossFadeOverlap} min={0} max={0.2} step={0.01} onChange={setCrossFadeOverlap} valueLabel={`${Math.round(crossFadeOverlap * 1000)} ms`} />
            <LabeledSlider label="Input gain" value={inputGain} min={0} max={2} step={0.01} onChange={setInputGain} valueLabel={inputGain.toFixed(2)} />
            <LabeledSlider label="Output gain" value={outputGain} min={0} max={2} step={0.01} onChange={setOutputGain} valueLabel={outputGain.toFixed(2)} />
          </div>

          <div className="mt-4 flex flex-wrap gap-1.5">
            {latencyPresets.map((preset) => (
              <MiniButton
                key={preset.id}
                onClick={() => onPreset(preset)}
                disabled={!statusLoaded || Boolean(busy)}
                active={readChunkSize === preset.chunk && crossFadeOverlap === preset.crossFade && extraConvert === preset.extra}
              >
                {preset.label}
              </MiniButton>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.15fr)_minmax(360px,0.85fr)]">
        <Panel title="Presets" aside={<Badge>{voicePresets.length}</Badge>}>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
            <input
              value={presetName}
              onChange={(event) => setPresetName(event.target.value)}
              placeholder={selectedPreset ? selectedPreset.name : "New preset name"}
              className={field}
            />
            <Button onClick={onSaveVoicePreset} disabled={!canApply || !presetName.trim()} className="whitespace-nowrap">
              {busy === "preset-save" ? "Saving..." : "Save As"}
            </Button>
            <Button onClick={onUpdateVoicePreset} disabled={!canApply || !selectedPreset} tone="primary" className="whitespace-nowrap">
              {busy === "preset-update" ? "Updating..." : "Update"}
            </Button>
          </div>

          <div className="mt-3 flex max-h-80 flex-col gap-2 overflow-y-auto pr-1">
            {voicePresets.length ? voicePresets.map((preset) => (
              <PresetCard
                key={preset.id}
                preset={preset}
                models={models}
                active={preset.id === selectedPresetId}
                canApply={canApply}
                busy={busy}
                onSelect={() => selectVoicePreset(preset.id)}
                onApply={() => void applyVoicePreset(preset)}
                onUpdate={onUpdateVoicePreset}
                onDelete={onDeleteVoicePreset}
              />
            )) : (
              <div className="rounded-md border border-white/10 bg-black/15 px-3 py-3 text-sm text-white/42">
                No saved voice presets yet.
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="Offline Convert"
          aside={(
            <Badge color={ready ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
              {ready ? "ready" : "not ready"}
            </Badge>
          )}
        >
          {offlineError ? (
            <div className="mb-3 rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">{offlineError}</div>
          ) : null}

          <div className="grid gap-3 xl:grid-cols-[minmax(220px,1fr)_minmax(220px,0.9fr)]">
            <label>
              <div className="mb-1.5 text-xs font-medium text-white/55">Audio file</div>
              <input
                type="file"
                accept=".wav,.flac,.ogg,.mp3,audio/wav,audio/flac,audio/ogg,audio/mpeg"
                onChange={(event) => setOfflineFile(event.target.files?.[0] ?? null)}
                className={`${field} file:mr-3 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-white/70`}
              />
            </label>
            <label>
              <div className="mb-1.5 text-xs font-medium text-white/55">Voice</div>
              <Select
                value={offlineModelId}
                onChange={setOfflineModelId}
                placeholder="no voices"
                options={voiceOptions}
                renderOption={(option) => <VoiceOption option={option} models={models} />}
              />
            </label>
          </div>

          <div className="mt-3 grid gap-3 xl:grid-cols-[1fr_1fr_auto]">
            <CompactSignedControl label="Pitch" value={offlinePitch} min={-24} max={24} step={1} onChange={(value) => setOfflinePitch(Math.round(value))} unit=" st" />
            <CompactSignedControl label="Formant" value={offlineFormant} min={-2} max={2} step={0.05} precision={2} onChange={setOfflineFormant} />
            <div className="flex items-end">
              <Button onClick={() => void onOfflineConvert()} disabled={!ready || !offlineFile || !offlineModelId || offlineBusy} tone="success" className="w-full xl:w-auto">
                {offlineBusy ? "Converting..." : "Convert"}
              </Button>
            </div>
          </div>

          {offlineResult ? (
            <div className="mt-4 rounded-md border border-white/10 bg-black/15 p-3">
              <audio controls src={offlineResult.url} className="w-full" />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm">
                <span className="flex gap-2">
                  <a href={offlineResult.url} download className="rounded-md border border-white/15 px-3 py-1.5 text-white/75 transition hover:bg-white/10 hover:text-white">WAV</a>
                  <a href={offlineResult.mp3_url} download className="rounded-md border border-white/15 px-3 py-1.5 text-white/75 transition hover:bg-white/10 hover:text-white">MP3</a>
                </span>
                <span className="text-xs text-white/45">
                  {offlineResult.sample_rate} Hz / {offlineResult.duration_s.toFixed(2)} s / pitch {offlineResult.params.pitch}
                  {" / "}formant {offlineResult.params.input_formant.toFixed(2)}
                  {" / "}denoise {offlineResult.params.input_denoise}
                </span>
              </div>
              <div className="mt-2 text-xs text-white/40">{timingsLine(offlineResult.timings_ms)}</div>
            </div>
          ) : null}
        </Panel>

        <Panel title="Diagnostics" aside={<Badge>{formatMs(status?.metrics.total_ms ?? status?.metrics.chunk_ms)}</Badge>}>
          <DiagnosticsCompact status={status} samples={meterHistory} />
        </Panel>
      </div>
    </div>
  );
}

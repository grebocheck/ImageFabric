import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { Badge } from "./Badge";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import { Meter, PerformanceBreakdown, WaveformMonitor, type MeterSample } from "./VoiceMeters";
import {
  DeviceSelect,
  LatencyMeter,
  MonitorSelect,
  OfflineDevice,
  RoutingApplyHint,
  Row,
  SetupStep,
  VoiceSlotList,
  type RoutingApplyState,
} from "./VoicePanelParts";
import {
  deviceName,
  f0Options,
  formatMs,
  latencyPresets,
  meter,
  nativeRoutingSettingsPatch,
  nativeSettingsToVoiceState,
  nativeTuningSettingsPatch,
  num,
  resolveMonitorDeviceId,
  sampleRates,
  selectedNativeModelId,
  waveformSlots,
} from "./voiceHelpers";
import type { VoiceEngineAsset, VoiceEngineConvertResult, VoiceEngineSettingsUpdate, VoiceEngineStatus } from "../types";

const field = "w-full rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5 text-sm outline-none focus:border-accent";
const assetSearchHint = "Place content_vec_500.onnx and rmvpe.pt in models/voice/pretrain.";
const modelDirHint = "models/voice";

const nativeF0Options = f0Options.map((option) => (
  option.value === "rmvpe"
    ? option
    : { ...option, disabled: true, hint: "not available in native mode" }
));

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
  return asset.found ? (asset.path ?? asset.name) : assetSearchHint;
}

function timingsLine(timings: Record<string, number>): string {
  const parts = Object.entries(timings)
    .filter(([, value]) => Number.isFinite(value))
    .map(([key, value]) => `${key} ${formatMs(value)}`);
  return parts.join(" / ") || "timings unavailable";
}

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceEngineStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [modelId, setModelId] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRatio, setIndexRatio] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Detector, setF0Detector] = useState("rmvpe");
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
  const [routingApplyState, setRoutingApplyState] = useState<RoutingApplyState>("idle");
  const [offlineFile, setOfflineFile] = useState<File | null>(null);
  const [offlineModelId, setOfflineModelId] = useState("");
  const [offlinePitch, setOfflinePitch] = useState(0);
  const [offlineBusy, setOfflineBusy] = useState(false);
  const [offlineError, setOfflineError] = useState("");
  const [offlineResult, setOfflineResult] = useState<VoiceEngineConvertResult | null>(null);
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
  const canApply = statusLoaded && !busy;
  const canGoLive = ready && Boolean(modelId) && !busy;

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

  useEffect(() => { void refresh(); }, [refresh]);

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
    setPitch(next.pitch);
    setOfflinePitch(next.pitch);
    setIndexRatio(next.indexRatio);
    setProtect(next.protect);
    setF0Detector(next.f0Detector);
    setPassThrough(next.passThrough);
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
  }, [modelId, status]);

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

  const tuningPatch = (): VoiceEngineSettingsUpdate => nativeTuningSettingsPatch({
    pitch,
    indexRatio,
    protect,
    f0Detector,
    passThrough,
  });

  const fullSettingsPatch = (): VoiceEngineSettingsUpdate => ({
    ...tuningPatch(),
    ...routingPatch,
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

  const onApply = () => run("apply", () => api.voiceEngineSettings(fullSettingsPatch()));

  const applyPatch = (label: string, patch: VoiceEngineSettingsUpdate) => run(label, () => api.voiceEngineSettings(patch));

  const onLive = (next: boolean) => run(next ? "live-on" : "live-off", async () => {
    if (!next) return api.voiceEngineSessionStop();
    if (!modelId) throw new Error("Select a voice model before starting live mode");
    await api.voiceEngineSettings(fullSettingsPatch());
    return api.voiceEngineSessionStart(modelId);
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
    if (statusLoaded) void applyPatch("bypass", { pass_through: next });
  };

  const onPtt = (next: boolean) => {
    setPtt(next);
    if (!statusLoaded) return;
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
  };

  const onOfflineConvert = async () => {
    if (!offlineFile) {
      setOfflineError("Choose a WAV, FLAC, or OGG file first");
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

  return (
    <div className="flex h-full w-full flex-col gap-4 overflow-y-auto p-1">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white/85">Voice changer</h2>
          <p className="mt-1 text-sm text-white/45">
            native RVC engine <span className="text-white/25">|</span> {live ? "live voice lane active" : "voice lane idle"}
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

      <div className="grid gap-3 xl:grid-cols-[1.15fr_1.05fr_2fr_1.8fr]">
        <SetupStep step="1" title="Engine" aside={(
          <Badge color={ready ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
            {ready ? "ready" : "missing"}
          </Badge>
        )}>
          <div className="flex flex-wrap gap-1.5">
            <Badge color={status?.stub ? "bg-sky-700/50 text-sky-100" : "bg-accent/50 text-accent-fg"}>
              {status?.stub ? "stub" : "real"}
            </Badge>
            <Badge>{status?.engine ?? "native-rvc"}</Badge>
          </div>
          <div className="mt-3 grid gap-1.5 text-sm">
            <Row label="Device" value={status?.device ?? "..."} ok={Boolean(status?.device)} mono />
            <Row label="Loaded" value={loadedModel?.name ?? status?.loaded_model ?? "not loaded"} ok={Boolean(status?.loaded_model)} />
          </div>
          <div className="mt-3 grid gap-1.5">
            {(status?.assets ?? []).map((asset) => (
              <div
                key={asset.name}
                title={assetTitle(asset)}
                className="flex items-center justify-between gap-2 rounded-md border border-white/10 bg-black/20 px-2.5 py-1.5"
              >
                <span className="min-w-0 truncate text-sm text-white/75">{asset.name}</span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <Badge color={asset.found ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
                    {asset.found ? "found" : "missing"}
                  </Badge>
                  {asset.source ? <Badge>{asset.source}</Badge> : null}
                </span>
              </div>
            ))}
            {!status?.assets?.length ? <div className="text-sm text-white/40">Loading native assets...</div> : null}
          </div>
          {!ready ? <p className="mt-2 text-xs leading-5 text-amber-100/70">{assetSearchHint}</p> : null}
        </SetupStep>

        <SetupStep step="2" title="Voice" aside={<Badge>{models.length}</Badge>}>
          <Select
            value={modelId}
            onChange={(value) => {
              setModelId(value);
              setOfflineModelId(value);
            }}
            placeholder="no voices"
            options={models.map((m) => ({ value: m.id, label: m.name, hint: `${m.source ?? "native"} #${m.slot}` }))}
          />
          <div className="mt-3 grid gap-1.5 text-sm">
            <Row label="Selected" value={selected?.name ?? "none"} />
            <Row label="Model" value={selected ? `${selected.type}${selected.version ? ` ${selected.version}` : ""}` : "none"} />
            <Row label="Index" value={selected?.has_index ? "available" : "none"} ok={selected?.has_index} />
          </div>
          <button
            type="button"
            onClick={() => setVoicesOpen((open) => !open)}
            className="mt-3 w-full rounded-md border border-white/10 px-2.5 py-1.5 text-left text-xs uppercase tracking-wide text-white/50 transition hover:bg-white/10 hover:text-white/75"
          >
            Voice slots
          </button>
          {voicesOpen ? (
            <VoiceSlotList models={models} modelId={modelId} modelDir={modelDirHint} onSelect={setModelId} />
          ) : null}
        </SetupStep>

        <SetupStep step="3" title="Audio devices" aside={(
          <RoutingApplyHint canReach={statusLoaded} state={routingApplyState} />
        )}>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-white/40">
            {statusLoaded ? `${inputDevices.length} inputs / ${outputDevices.length} outputs` : "Loading native device list"}
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {inputDevices.length ? (
              <DeviceSelect
                label="Input"
                value={inputDeviceId}
                devices={inputDevices}
                fallback="No input selected"
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
                onChange={setOutputDeviceId}
              />
            ) : (
              <OfflineDevice label="Output" message={statusLoaded ? "No output devices reported" : "Loading devices"} />
            )}
            {outputDevices.length ? (
              <MonitorSelect
                value={monitorDeviceId}
                devices={outputDevices}
                onChange={setMonitorDeviceId}
              />
            ) : (
              <OfflineDevice label="Monitor" message={statusLoaded ? "No output device for monitor" : "Loading devices"} />
            )}
          </div>
        </SetupStep>

        <SetupStep step="4" title="Go live" aside={(
          <Badge color={live ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
            {live ? "live" : "off"}
          </Badge>
        )}>
          <div className="grid gap-3">
            {status?.session_error ? (
              <div className="rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">
                {status.session_error}
              </div>
            ) : null}

            <div className={`rounded-md border px-3 py-3 ${live ? "border-emerald-400/30 bg-emerald-400/10" : "border-white/10 bg-black/20"}`}>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-white/80">{live ? "Live voice active" : "Live voice off"}</div>
                  <div className="mt-0.5 text-xs text-white/35">
                    {live
                      ? `${selected?.name ?? "voice"}: mic -> converted output${monitorOn ? " + monitor" : ""}`
                      : ready ? "native engine ready" : "missing assets or voice model"}
                  </div>
                </div>
              </div>
              {live ? (
                <button
                  onClick={() => onLive(false)}
                  disabled={Boolean(busy)}
                  className="mt-3 w-full rounded-md bg-red-600/90 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-red-500 disabled:opacity-40"
                >
                  {busy === "live-off" ? "Stopping..." : "Stop live voice"}
                </button>
              ) : (
                <button
                  onClick={() => onLive(true)}
                  disabled={!canGoLive}
                  className="mt-3 w-full rounded-md bg-emerald-600 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-40 disabled:hover:bg-emerald-600"
                >
                  {busy === "live-on" ? "Starting..." : "Start live voice"}
                </button>
              )}
              {!live && !canGoLive ? (
                <div className="mt-1.5 text-xs text-amber-200/75">
                  {busy
                    ? "busy..."
                    : !ready
                      ? "engine not ready: check the assets in step 1"
                      : !modelId
                        ? "select a voice in step 2 first"
                        : "cannot start right now"}
                </div>
              ) : null}
            </div>

            <div className={`rounded-md border px-3 py-2 ${monitorOn ? "border-sky-400/30 bg-sky-400/10" : "border-white/10 bg-black/20"}`}>
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white/80">Monitor</span>
                    <Badge color={monitorOn ? "bg-sky-700/50 text-sky-100" : "bg-white/10 text-white/55"}>
                      {monitorOn ? "on" : "off"}
                    </Badge>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-white/35" title={deviceName(outputDevices, monitorDeviceId, "Off")}>
                    {deviceName(outputDevices, monitorDeviceId, "Off")}
                  </div>
                </div>
                <Toggle checked={monitorOn} onChange={onMonitor} disabled={!statusLoaded || outputDevices.length === 0} ariaLabel="Toggle monitor" />
              </div>
              <div className="mt-2">
                <div className="text-xs uppercase tracking-wide text-white/40">Monitor gain</div>
                <Slider value={monitorGain} min={0} max={2} step={0.01} onChange={setMonitorGain} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <Meter label="Input" value={meter(status?.metrics.input_vu ?? 0)} />
              <Meter label={monitorOn ? "Output / monitor" : "Output"} value={meter(status?.metrics.output_vu ?? 0)} tone="sky" />
              <LatencyMeter value={status?.metrics.total_ms ?? status?.metrics.chunk_ms} />
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Badge>overruns {status?.metrics.overruns ?? 0}</Badge>
              <Badge>underruns {status?.metrics.underruns ?? 0}</Badge>
              <Badge>chunk {formatMs(status?.metrics.chunk_ms)}</Badge>
            </div>
          </div>
        </SetupStep>
      </div>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-white/40">Tuning</div>
            <div className="mt-1 text-xs text-white/35">{selected?.name ?? "no voice selected"}</div>
          </div>
          <button
            onClick={onApply}
            disabled={!canApply}
            className="rounded-md border border-white/15 px-3 py-1.5 text-sm font-medium text-white/75 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
          >
            {busy === "apply" ? "Applying..." : "Apply tuning"}
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">F0 detector</div>
            <Select value={f0Detector} onChange={setF0Detector} className="mt-1" options={nativeF0Options} />
            <div className="mt-1 text-[11px] text-white/35">Only RMVPE is implemented in native mode.</div>
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

          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Input gain</div>
            <Slider value={inputGain} min={0} max={2} step={0.01} onChange={setInputGain} />
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Output gain</div>
            <Slider value={outputGain} min={0} max={2} step={0.01} onChange={setOutputGain} />
          </div>

          <div className="flex items-end justify-between gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2">
            <label className="flex items-center gap-2 text-xs text-white/55">
              <Toggle checked={passThrough} onChange={onBypass} disabled={!statusLoaded || Boolean(busy)} />
              Bypass
            </label>
            <label className="flex items-center gap-2 text-xs text-white/55">
              <Toggle checked={ptt} onChange={onPtt} disabled={!statusLoaded} />
              PTT
            </label>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-md border border-white/10 bg-black/20 px-3 py-2">
          <div className="text-xs uppercase tracking-wide text-white/40">Latency presets</div>
          <div className="flex gap-1.5">
            {latencyPresets.map((preset) => (
              <button
                key={preset.id}
                onClick={() => onPreset(preset)}
                disabled={!statusLoaded || Boolean(busy)}
                className={`rounded border px-2 py-1 text-xs transition disabled:opacity-30 ${
                  readChunkSize === preset.chunk && crossFadeOverlap === preset.crossFade && extraConvert === preset.extra
                    ? "border-accent/40 bg-accent/15 text-white"
                    : "border-white/10 text-white/65 hover:bg-white/10"
                }`}
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
            <div className="text-xs font-medium uppercase tracking-wide text-white/40">Audio engine</div>
            <div className="mt-1 text-xs text-white/35">
              <RoutingApplyHint canReach={statusLoaded} state={routingApplyState} />
            </div>
          </div>
          <Badge color={live ? "bg-emerald-700/55 text-emerald-100" : "bg-white/10 text-white/55"}>
            {live ? "live" : "stopped"}
          </Badge>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
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

          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Extra buffer</div>
            <Slider value={extraConvert} min={0} max={10} step={0.1} onChange={setExtraConvert} />
          </div>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Crossfade</div>
            <Slider value={crossFadeOverlap} min={0} max={0.2} step={0.01} onChange={setCrossFadeOverlap} />
          </div>
          <div className="grid gap-2 rounded-md border border-white/10 bg-black/20 px-3 py-2 text-sm md:grid-cols-3">
            <Row label="Input" value={deviceName(inputDevices, inputDeviceId, "None")} />
            <Row label="Output" value={deviceName(outputDevices, outputDeviceId, "None")} />
            <Row label="Monitor" value={deviceName(outputDevices, monitorDeviceId, "Off")} />
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-white/40">Offline convert</div>
            <div className="mt-1 text-xs text-white/35">WAV, FLAC, or OGG through the native RVC pipeline</div>
          </div>
          <Badge color={ready ? "bg-emerald-700/55 text-emerald-100" : "bg-amber-600/40 text-amber-100"}>
            {ready ? "ready" : "not ready"}
          </Badge>
        </div>

        {offlineError ? (
          <div className="mb-3 rounded-md border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">{offlineError}</div>
        ) : null}

        <div className="grid gap-3 md:grid-cols-[1.3fr_1fr_0.5fr_auto]">
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Audio file</div>
            <input
              type="file"
              accept=".wav,.flac,.ogg,audio/wav,audio/flac,audio/ogg"
              onChange={(event) => setOfflineFile(event.target.files?.[0] ?? null)}
              className={`${field} mt-1 file:mr-3 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1 file:text-xs file:text-white/70`}
            />
          </label>
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Voice</div>
            <Select
              value={offlineModelId}
              onChange={setOfflineModelId}
              placeholder="no voices"
              className="mt-1"
              options={models.map((m) => ({ value: m.id, label: m.name, hint: m.source ?? "" }))}
            />
          </label>
          <label>
            <div className="text-xs uppercase tracking-wide text-white/40">Pitch</div>
            <input
              type="number"
              min={-24}
              max={24}
              step={1}
              value={offlinePitch}
              onChange={(event) => setOfflinePitch(Number(event.target.value))}
              className={`${field} mt-1`}
            />
          </label>
          <div className="flex items-end">
            <button
              onClick={() => void onOfflineConvert()}
              disabled={!ready || !offlineFile || !offlineModelId || offlineBusy}
              className="w-full rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-30 disabled:hover:bg-emerald-600"
            >
              {offlineBusy ? "Converting..." : "Convert"}
            </button>
          </div>
        </div>

        {offlineResult ? (
          <div className="mt-4 rounded-md border border-white/10 bg-black/20 p-3">
            <audio controls src={offlineResult.url} className="w-full" />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm">
              <a
                href={offlineResult.url}
                download
                className="rounded-md border border-white/15 px-3 py-1.5 text-white/75 transition hover:bg-white/10 hover:text-white"
              >
                Download WAV
              </a>
              <span className="text-xs text-white/45">
                {offlineResult.sample_rate} Hz / {offlineResult.duration_s.toFixed(2)} s / pitch {offlineResult.params.pitch}
              </span>
            </div>
            <div className="mt-2 text-xs text-white/40">{timingsLine(offlineResult.timings_ms)}</div>
          </div>
        ) : null}
      </section>

      <section className="rounded-lg border border-white/10 bg-surface p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-xs font-medium uppercase tracking-wide text-white/40">Diagnostics</div>
          <Badge>{formatMs(status?.metrics.total_ms ?? status?.metrics.chunk_ms)}</Badge>
        </div>
        <div className="grid gap-3 lg:grid-cols-[1.2fr_0.8fr]">
          <WaveformMonitor samples={meterHistory} />
          <PerformanceBreakdown metrics={status?.metrics} />
        </div>
      </section>
    </div>
  );
}

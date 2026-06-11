import type { VoiceAudioDevice, VoiceModel, VoiceSettingsUpdate } from "../types";

export const f0Options = [
  { value: "rmvpe_onnx", label: "RMVPE ONNX" },
  { value: "rmvpe", label: "RMVPE" },
  { value: "crepe_onnx_tiny", label: "CREPE tiny ONNX" },
  { value: "crepe_onnx_full", label: "CREPE full ONNX" },
  { value: "crepe_tiny", label: "CREPE tiny" },
  { value: "crepe_full", label: "CREPE full" },
  { value: "fcpe", label: "FCPE" },
  { value: "fcpe_onnx", label: "FCPE ONNX" },
];

export const sampleRates = [16000, 24000, 44100, 48000, 96000];

export const latencyPresets = [
  { id: "fast", label: "Fast", chunk: 96, crossFade: 0.03, extra: 3 },
  { id: "balanced", label: "Balanced", chunk: 133, crossFade: 0.05, extra: 5 },
  { id: "quality", label: "Quality", chunk: 192, crossFade: 0.08, extra: 7 },
] as const;

export const waveformSlots = 64;
export const timingLabels = ["prep", "f0", "infer", "post", "io", "mix"];

export type VoiceControlState = {
  pitch: number;
  formantShift: number;
  indexRatio: number;
  protect: number;
  f0Detector: string;
  passThrough: boolean;
  inputDeviceId: number;
  outputDeviceId: number;
  monitorDeviceId: number;
  sampleRate: number;
  readChunkSize: number;
  crossFadeOverlap: number;
  extraConvert: number;
  inputGain: number;
  outputGain: number;
  monitorGain: number;
};

export type VoiceRoutingState = Pick<
  VoiceControlState,
  | "inputDeviceId"
  | "outputDeviceId"
  | "monitorDeviceId"
  | "sampleRate"
  | "readChunkSize"
  | "crossFadeOverlap"
  | "extraConvert"
  | "inputGain"
  | "outputGain"
  | "monitorGain"
>;

export function num(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function settingsToVoiceState(settings: Record<string, unknown>): VoiceControlState {
  const f0 = String(settings.f0Detector ?? "rmvpe_onnx");
  return {
    pitch: num(settings.tran, 0),
    formantShift: num(settings.formantShift, 0),
    indexRatio: num(settings.indexRatio, 1),
    protect: num(settings.protect, 0.5),
    f0Detector: f0Options.some((o) => o.value === f0) ? f0 : "rmvpe_onnx",
    passThrough: Boolean(settings.passThrough),
    inputDeviceId: num(settings.serverInputDeviceId, -1),
    outputDeviceId: num(settings.serverOutputDeviceId, -1),
    monitorDeviceId: num(settings.serverMonitorDeviceId, -1),
    sampleRate: num(settings.serverAudioSampleRate, 48000),
    readChunkSize: num(settings.serverReadChunkSize, 133),
    crossFadeOverlap: num(settings.crossFadeOverlapSize, 0.05),
    extraConvert: num(settings.extraConvertSize, 5),
    inputGain: num(settings.serverInputAudioGain, 1),
    outputGain: num(settings.serverOutputAudioGain, 1),
    monitorGain: num(settings.serverMonitorAudioGain, 1),
  };
}

export function routingSettingsPatch(state: VoiceRoutingState): VoiceSettingsUpdate {
  return {
    server_input_device_id: state.inputDeviceId,
    server_output_device_id: state.outputDeviceId,
    server_monitor_device_id: state.monitorDeviceId,
    server_audio_sample_rate: state.sampleRate,
    server_read_chunk_size: state.readChunkSize,
    cross_fade_overlap_size: state.crossFadeOverlap,
    extra_convert_size: state.extraConvert,
    server_input_gain: state.inputGain,
    server_output_gain: state.outputGain,
    server_monitor_gain: state.monitorGain,
  };
}

export function selectedModelId(models: VoiceModel[], selectedSlot: string | null): string {
  return models.find((m) => m.slot === selectedSlot)?.id ?? models[0]?.id ?? "";
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}

export function perfSummary(performance: Record<string, unknown> | null): string {
  if (!performance) return "...";
  const entries = Object.entries(performance)
    .filter(([, value]) => ["number", "string", "boolean"].includes(typeof value))
    .slice(0, 3)
    .map(([key, value]) => `${key}:${String(value)}`);
  return entries.join(", ") || "available";
}

export function deviceHint(hostApi: string, rate: number | null): string {
  return [hostApi, rate ? `${rate / 1000}k` : ""].filter(Boolean).join(", ");
}

export function deviceNumericId(device: Pick<VoiceAudioDevice, "id" | "index">): number {
  return num(device.id, device.index);
}

export function findDevice(devices: Pick<VoiceAudioDevice, "id" | "index">[], id: number): Pick<VoiceAudioDevice, "id" | "index"> | undefined {
  return devices.find((device) => deviceNumericId(device) === id);
}

export function deviceName(devices: Pick<VoiceAudioDevice, "id" | "index" | "name">[], id: number, fallback = "Not selected"): string {
  return devices.find((device) => deviceNumericId(device) === id)?.name ?? fallback;
}

export function resolveMonitorDeviceId(
  currentMonitorDeviceId: number,
  selectedOutputDeviceId: number,
  outputDevices: Pick<VoiceAudioDevice, "id" | "index">[],
): number {
  if (currentMonitorDeviceId >= 0) return currentMonitorDeviceId;
  if (selectedOutputDeviceId >= 0 && findDevice(outputDevices, selectedOutputDeviceId)) return selectedOutputDeviceId;
  return outputDevices[0] ? deviceNumericId(outputDevices[0]) : -1;
}

export function meter(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "...";
  return `${Number(value).toFixed(1)} ms`;
}

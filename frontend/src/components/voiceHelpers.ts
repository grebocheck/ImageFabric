import type { VoiceAudioDevice, VoiceEngineSettings, VoiceEngineSettingsUpdate, VoiceModel } from "../types";

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
export const inputHighpassOptions = [
  { value: "0", label: "Off" },
  { value: "60", label: "60 Hz" },
  { value: "80", label: "80 Hz" },
  { value: "120", label: "120 Hz" },
  { value: "160", label: "160 Hz" },
];

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
  inputGateDb: number;
  inputHighpassHz: number;
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

export function nativeSettingsToVoiceState(settings: Partial<Record<keyof VoiceEngineSettings, unknown>>): VoiceControlState {
  const f0 = String(settings.f0_detector ?? "rmvpe");
  return {
    pitch: num(settings.pitch, 0),
    formantShift: num(settings.input_formant, 0),
    inputGateDb: num(settings.input_gate_db, -60),
    inputHighpassHz: num(settings.input_highpass_hz, 80),
    indexRatio: num(settings.index_ratio, 1),
    protect: num(settings.protect, 0.5),
    f0Detector: f0Options.some((o) => o.value === f0) ? f0 : "rmvpe",
    passThrough: Boolean(settings.pass_through),
    inputDeviceId: settings.server_input_device_id == null ? -1 : num(settings.server_input_device_id, -1),
    outputDeviceId: settings.server_output_device_id == null ? -1 : num(settings.server_output_device_id, -1),
    monitorDeviceId: settings.server_monitor_device_id == null ? -1 : num(settings.server_monitor_device_id, -1),
    sampleRate: num(settings.server_audio_sample_rate, 48000),
    readChunkSize: num(settings.server_read_chunk_size, 133),
    crossFadeOverlap: num(settings.cross_fade_overlap_size, 0.05),
    extraConvert: num(settings.extra_convert_size, 2),
    inputGain: num(settings.server_input_gain, 1),
    outputGain: num(settings.server_output_gain, 1),
    monitorGain: num(settings.server_monitor_gain, 1),
  };
}

export function nativeRoutingSettingsPatch(state: VoiceRoutingState): VoiceEngineSettingsUpdate {
  return {
    server_input_device_id: state.inputDeviceId >= 0 ? state.inputDeviceId : null,
    server_output_device_id: state.outputDeviceId >= 0 ? state.outputDeviceId : null,
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

export function nativeTuningSettingsPatch(state: Pick<
  VoiceControlState,
  | "pitch"
  | "formantShift"
  | "inputGateDb"
  | "inputHighpassHz"
  | "indexRatio"
  | "protect"
  | "f0Detector"
  | "passThrough"
>): VoiceEngineSettingsUpdate {
  return {
    pitch: state.pitch,
    input_formant: state.formantShift,
    input_gate_db: state.inputGateDb,
    input_highpass_hz: state.inputHighpassHz,
    index_ratio: state.indexRatio,
    protect: state.protect,
    f0_detector: state.f0Detector,
    pass_through: state.passThrough,
  };
}

export function selectedNativeModelId(models: VoiceModel[], current: string, loaded: string | null | undefined): string {
  if (current && models.some((m) => m.id === current)) return current;
  if (loaded && models.some((m) => m.id === loaded)) return loaded;
  return models[0]?.id ?? "";
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
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

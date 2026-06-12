export type JobType = "llm" | "image";
export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";
export type ModelFamily = "flux" | "flux2" | "qwen-image" | "z-image" | "sdxl" | "gguf" | "unknown";
export type AppTheme = "dark" | "dim" | "light";

export interface Model {
  id: string;
  name: string;
  family: ModelFamily;
  job_type: JobType;
  size_bytes: number;
  loaded: boolean;
  warm?: boolean;
  quant?: string | null;
  estimated_vram_gb?: number | null;
  vram_measured?: boolean;
  slow?: boolean;
}

export interface WarmModel {
  resident: string;
  model_id: string;
  model: string;
  family: string;
}

export interface Lora {
  id: string;
  name: string;
  family: ModelFamily | null;
  size_bytes: number;
}

export interface GpuStatus {
  resident: string | null;
  model_id: string | null;
  model: string | null;
  family: string | null;
  warm?: WarmModel[];
}

export interface RamStats {
  total_gb: number;
  available_gb: number;
  used_gb: number;
  percent: number;
  process_rss_gb: number;
}

export interface VramStats {
  total_gb: number;
  free_gb: number;
  used_gb: number;
}

export interface MemSnapshot {
  ram: RamStats | null;
  vram: VramStats | null;
}

export interface SecurityPosture {
  exposed: boolean;
  token_required: boolean;
}

export interface HealthStatus {
  status: string;
  stub_mode: boolean;
  models: number;
  gpu: GpuStatus;
  mem: MemSnapshot;
  security: SecurityPosture;
}

// One point in the rolling memory-pressure timeline (System tab).
export interface MemPoint {
  ts: number;
  ram: RamStats | null;
  vram: VramStats | null;
  resident: string | null;
}

// Predicted scheduler drain order for the current queue (P7.4).
export interface QueuePlanStep {
  model_id: string;
  model: string;
  type: JobType;
  count: number;
}

export interface QueuePlan {
  queued: number;
  swaps: number;
  current_model_id: string | null;
  current_model: string | null;
  steps: QueuePlanStep[];
}

// A structured reason the arbiter held / swapped / refused a load (P7.1).
export interface ArbiterNote {
  reason: string;
  message: string;
  model?: string;
  family?: string;
  predicted_gb?: number;
  available_gb?: number;
  ts: number;
}

export interface RuntimeSettings {
  stub_mode: boolean;
  paths: Record<string, string>;
  memory: Record<string, unknown>;
  acceleration: Record<string, unknown>;
  counts: Record<string, number>;
  gpu: GpuStatus;
  mem: Record<string, unknown>;
}

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  priority: number;
  model_id: string;
  params: Record<string, unknown>;
  progress: number;
  result: Record<string, unknown> | null;
  error: string | null;
  progress_note?: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ImageItem {
  id: string;
  job_id: string;
  seed: number | null;
  width: number | null;
  height: number | null;
  family: ModelFamily | "unknown";
  favorite: boolean;
  tags: string[];
  params: Record<string, unknown>;
  created_at: string;
  url: string;
  thumb_url: string | null;
}

export interface ImageStats {
  total: number;
  today: number;
  by_model: { model: string; count: number }[];
  by_family?: { family: ModelFamily | "unknown"; count: number }[];
  by_lora?: { id: string; name: string; count: number }[];
  by_tag?: { tag: string; count: number }[];
}

// A request to load params into the image composer (from History / a result).
export interface ComposerApply {
  model_id?: string;
  params: Record<string, unknown>;
  nonce: number;
}

export interface Preset {
  id: string;
  name: string;
  type: JobType;
  params: Record<string, unknown>;
  created_at: string;
}

export interface JobCreate {
  type: JobType;
  model_id: string;
  params: Record<string, unknown>;
  priority?: number;
}

export interface BusEvent {
  type: string;
  ts: number;
  [k: string]: unknown;
}

export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  error?: boolean;
  job_id?: string | null;
  created_at?: string;
}

export interface ChatConversation {
  id: string;
  title: string;
  model_id: string | null;
  system: string | null;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ChatConversationDetail extends ChatConversation {
  messages: ChatMessage[];
}

export interface ChatImportMessage {
  role: ChatRole;
  content: string;
  error?: boolean;
  created_at?: string;
}

export interface ChatConversationImport {
  title?: string;
  model_id?: string | null;
  system?: string | null;
  params?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  messages?: ChatImportMessage[];
}

export interface ChatImportResult {
  imported: number;
  conversations: ChatConversationDetail[];
}

export interface ChatSendResult {
  job_id: string;
  conversation: ChatConversation;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface ChatSendBody {
  content: string;
  model_id: string;
  system?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repeat_penalty?: number;
  seed?: number;
  stop?: string[];
  image_tool?: boolean;
  image_model_id?: string;
  document_tool?: boolean;
  rag_top_k?: number;
}

export interface LlmContextType {
  id: string;
  label: string;
  experimental: boolean;
}

export interface LlmBackendInfo {
  id: string;
  label: string;
  available: boolean;
  path: string;
  context_types: string[];
}

export interface LlmConfig {
  ctx: number;
  ngl: number;
  backend: string;
  backends: LlmBackendInfo[];
  context_type: string;
  context_types: LlmContextType[];
  stub: boolean;
  loaded: boolean;
  model_id: string | null;
  defaults: { temperature: number; max_tokens: number };
}

export interface PresetImportItem {
  name: string;
  type: JobType;
  params: Record<string, unknown>;
}

export interface PresetImportResult {
  imported: number;
  skipped: number;
  presets: Preset[];
}

export interface Note {
  id: string;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface TtsModel {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
}

export interface TtsStatus {
  binary: string;
  binary_exists: boolean;
  models_dir: string;
  models: TtsModel[];
  ready: boolean;
}

export interface TtsGenerateBody {
  model_id: string;
  text: string;
  vocoder_id?: string | null;
  use_guide_tokens?: boolean;
}

export interface TtsGenerateResult {
  id: string;
  url: string;
  path: string;
  metadata_path: string;
  model_id: string;
  vocoder_id?: string | null;
  duration_seconds: number;
}

export interface TranscriptionModel {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
  engine: "faster-whisper" | "openai-whisper";
}

export interface TranscriptionStatus {
  models_dir: string;
  models: TranscriptionModel[];
  engines: Record<string, boolean>;
  device: string;
  compute_type: string;
  max_upload_mb: number;
  ready: boolean;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscriptionResult {
  id: string;
  text: string;
  segments: TranscriptionSegment[];
  detected_language?: string | null;
  language_probability?: number | null;
  metadata_url: string;
  metadata_path: string;
  duration_seconds: number;
}

export interface RagModel {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
}

export interface RagStatus {
  binary: string;
  binary_exists: boolean;
  models_dir: string;
  models: RagModel[];
  ready: boolean;
  port: number;
  gpu_layers: number;
  chunk_chars: number;
  chunk_overlap: number;
}

export interface RagDocument {
  id: string;
  title: string;
  source?: string | null;
  model_id?: string | null;
  chunks_count: number;
  created_at: string;
  updated_at: string;
}

export interface RagSearchResult {
  document_id: string;
  document_title: string;
  chunk_id: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface RagSearchResponse {
  query: string;
  results: RagSearchResult[];
  context: string;
}

export interface VisionModel {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
}

export interface VisionStatus {
  binary: string;
  binary_exists: boolean;
  models_dir: string;
  models: VisionModel[];
  projectors: VisionModel[];
  ready: boolean;
  gpu_layers: number;
  max_upload_mb: number;
}

export interface VisionResult {
  id: string;
  text: string;
  metadata_url: string;
  metadata_path: string;
  duration_seconds: number;
}

export interface VoiceModel {
  id: string;
  slot: string;
  name: string;
  type: string;
  version: string;
  sampling_rate: number | null;
  f0: boolean;
  has_index: boolean;
  size_bytes: number;
  source?: string;
}

export interface VoiceAudioDevice {
  id: string;
  index: number;
  name: string;
  host_api: string;
  max_input_channels: number;
  max_output_channels: number;
  default_sample_rate: number | null;
}

export interface VoiceEngineAsset {
  name: string;
  path: string | null;
  found: boolean;
  source: string | null;
}

export interface VoiceEngineSettings {
  pitch: number;
  index_ratio: number;
  protect: number;
  f0_detector: string;
  server_input_device_id: number | null;
  server_output_device_id: number | null;
  server_monitor_device_id: number | null;
  server_input_gain: number;
  server_output_gain: number;
  server_monitor_gain: number;
  server_audio_sample_rate: number;
  server_read_chunk_size: number;
  cross_fade_overlap_size: number;
  extra_convert_size: number;
  pass_through: boolean;
}

export interface VoiceEngineSettingsUpdate {
  pitch?: number | null;
  index_ratio?: number | null;
  protect?: number | null;
  f0_detector?: string | null;
  server_input_device_id?: number | null;
  server_output_device_id?: number | null;
  server_monitor_device_id?: number | null;
  server_input_gain?: number | null;
  server_output_gain?: number | null;
  server_monitor_gain?: number | null;
  server_audio_sample_rate?: number | null;
  server_read_chunk_size?: number | null;
  cross_fade_overlap_size?: number | null;
  extra_convert_size?: number | null;
  pass_through?: boolean | null;
}

export interface VoiceEngineMetrics {
  input_vu: number;
  output_vu: number;
  timings_ms: Record<string, number>;
  total_ms: number | null;
  chunk_ms: number | null;
  overruns: number;
  underruns: number;
}

export interface VoiceEngineStatus {
  engine: string;
  stub: boolean;
  ready: boolean;
  assets: VoiceEngineAsset[];
  models: VoiceModel[];
  audio_devices: {
    inputs: VoiceAudioDevice[];
    outputs: VoiceAudioDevice[];
  };
  device: string;
  settings: VoiceEngineSettings;
  loaded_model: string | null;
  live: boolean;
  session_error: string | null;
  metrics: VoiceEngineMetrics;
}

export interface VoiceEngineConvertResult {
  token: string;
  url: string;
  duration_s: number;
  sample_rate: number;
  timings_ms: Record<string, number>;
  model_id: string;
  params: {
    pitch: number;
    index_ratio: number;
    protect: number;
    f0_detector: string;
  };
}

export interface CodeFile {
  path: string;
  size_bytes: number;
}

export interface CodeFileContent extends CodeFile {
  content: string;
  truncated: boolean;
}

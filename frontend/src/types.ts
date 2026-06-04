export type JobType = "llm" | "image";
export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";
export type ModelFamily = "flux" | "flux2" | "sdxl" | "gguf" | "unknown";

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
  params: Record<string, unknown>;
  created_at: string;
  url: string;
  thumb_url: string | null;
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
}

export interface LlmConfig {
  ctx: number;
  ngl: number;
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

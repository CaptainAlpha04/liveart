// Shared TypeScript interfaces that mirror the backend Pydantic schemas in
// `backend/schemas/__init__.py`. Keep these in lockstep with the spec
// (see docs/superpowers/specs/2026-04-18-liveart-design.md §4 and §5).

export interface StyleInfo {
  id: string;
  name: string;
  artist: string;
  is_pretrained: boolean;
  thumbnail_url: string;
}

// A curated reference artwork available in backend/style_sources/. Distinct
// from StyleInfo — StyleSource is training *input*; StyleInfo is a trained
// *model* that can be used for inference.
export interface StyleSource {
  id: string;
  name: string;
  artist: string;
  image_url: string;
}

export interface TrainingConfig {
  style_name: string;
  style_weight: number;
  content_weight: number;
  tv_weight: number;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  image_size: number;
}

export type TrainingState = "idle" | "running" | "done" | "error";

export interface TrainingStatus {
  state: TrainingState;
  style_name: string | null;
  style_id: string | null;
  epoch: number;
  batch: number;
  total_batches: number;
  progress: number;
  error?: string | null;
}

// Emitted every 50 batches by the backend training loop. Optional fields
// (status, model_id, elapsed_s) are present for the terminal completion
// message; the spec allows a second message shape for "done".
// Events arrive in two shapes:
//   • Progress events (every 50 batches while status === "running"):
//     carry all loss values, epoch/batch counters, elapsed_s and eta_s.
//   • Terminal events ({status: "done" | "error" | "stopped"}):
//     carry only model_id / elapsed_s / error — the loss fields are absent.
// Everything except ``status`` and ``batch`` is marked optional to reflect this.
export interface TrainingEvent {
  epoch?: number;
  batch?: number;
  total_batches?: number;
  content_loss?: number;
  style_loss?: number;
  tv_loss?: number;
  total_loss?: number;
  elapsed_s?: number;
  eta_s?: number;
  status?: "running" | "done" | "error" | "stopped";
  model_id?: string;
  style_id?: string;
  error?: string;
}

export type VideoJobState = "queued" | "processing" | "done" | "error";

export interface VideoJobStatus {
  job_id: string;
  status: VideoJobState;
  progress: number;
  total_frames: number;
  processed_frames: number;
  elapsed_s: number;
  error?: string | null;
}

export interface InferenceStats {
  fps: number;
  inferenceMs: number;
  style: string;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "error";
  gpu_available: boolean;
  gpu_name: string | null;
  models_loaded: number;
  uptime_s: number;
}

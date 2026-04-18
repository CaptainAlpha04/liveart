// Typed fetch wrappers for the REST surface. All paths are relative so that
// the Vite dev-server proxy (see frontend/vite.config.ts) can forward them to
// the FastAPI backend on :8000. In production the frontend is served from the
// same origin as the backend, so relative paths continue to work.

import type {
  HealthStatus,
  StyleInfo,
  StyleSource,
  TrainingConfig,
  TrainingStatus,
  VideoJobStatus,
} from "../types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function fetchStyles(): Promise<StyleInfo[]> {
  return json<StyleInfo[]>(await fetch("/api/styles"));
}

export async function fetchModels(): Promise<StyleInfo[]> {
  return json<StyleInfo[]>(await fetch("/api/models"));
}

export async function refreshModels(): Promise<StyleInfo[]> {
  return json<StyleInfo[]>(
    await fetch("/api/models/refresh", { method: "POST" })
  );
}

export async function deleteModel(id: string): Promise<void> {
  const res = await fetch(`/api/models/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
}

export interface VideoUploadResponse {
  job_id: string;
}

export async function uploadVideo(
  file: File,
  styleId: string
): Promise<VideoUploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("style_id", styleId);
  return json<VideoUploadResponse>(
    await fetch("/api/video/stylize", { method: "POST", body: fd })
  );
}

export async function videoStatus(id: string): Promise<VideoJobStatus> {
  return json<VideoJobStatus>(
    await fetch(`/api/video/status/${encodeURIComponent(id)}`)
  );
}

export function videoDownloadUrl(id: string): string {
  return `/api/video/download/${encodeURIComponent(id)}`;
}

export interface StartTrainingResponse {
  job_id: string;
}

export async function fetchStyleSources(): Promise<StyleSource[]> {
  return json<StyleSource[]>(await fetch("/api/training/style-sources"));
}

/**
 * Start a training run from an uploaded style image.
 */
export async function startTrainingFromUpload(
  imageFile: File,
  config: TrainingConfig
): Promise<StartTrainingResponse> {
  const fd = new FormData();
  fd.append("style_image", imageFile);
  fd.append("config", JSON.stringify(config));
  return json<StartTrainingResponse>(
    await fetch("/api/training/start", { method: "POST", body: fd })
  );
}

/**
 * Start a training run using one of the curated reference artworks shipped in
 * ``backend/style_sources/``.
 */
export async function startTrainingFromSource(
  styleSourceId: string,
  config: TrainingConfig
): Promise<StartTrainingResponse> {
  const fd = new FormData();
  fd.append("style_source_id", styleSourceId);
  fd.append("config", JSON.stringify(config));
  return json<StartTrainingResponse>(
    await fetch("/api/training/start", { method: "POST", body: fd })
  );
}

// Backward-compatible alias — existing callers still work.
export const startTraining = startTrainingFromUpload;

export async function stopTraining(): Promise<{ status: string }> {
  return json<{ status: string }>(
    await fetch("/api/training/stop", { method: "POST" })
  );
}

export async function trainingStatus(): Promise<TrainingStatus> {
  return json<TrainingStatus>(await fetch("/api/training/status"));
}

export async function health(): Promise<HealthStatus> {
  return json<HealthStatus>(await fetch("/health"));
}

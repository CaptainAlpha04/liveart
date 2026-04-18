import type { FormEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  uploadVideo,
  videoDownloadUrl,
  videoStatus,
} from "../api/client";
import type { VideoJobStatus } from "../types";

interface VideoUploadProps {
  styleId: string;
}

// File-based video pipeline: pick a .mp4, POST it to /api/video/stylize,
// poll /api/video/status every 500ms while processing, then reveal a
// download link pointing at /api/video/download/{id}.
export function VideoUpload({ styleId }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<VideoJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setError(null);
      if (!file) {
        setError("Select a video file first.");
        return;
      }
      if (!styleId) {
        setError("Pick a style first.");
        return;
      }
      setSubmitting(true);
      setStatus(null);
      try {
        const { job_id } = await uploadVideo(file, styleId);
        setJobId(job_id);
        // Begin polling
        stopPolling();
        pollRef.current = setInterval(async () => {
          try {
            const s = await videoStatus(job_id);
            setStatus(s);
            if (s.status === "done" || s.status === "error") {
              stopPolling();
            }
          } catch (err) {
            setError(err instanceof Error ? err.message : "Poll failed");
            stopPolling();
          }
        }, 500);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setSubmitting(false);
      }
    },
    [file, styleId, stopPolling]
  );

  const percent = status ? Math.round(status.progress * 100) : 0;
  const done = status?.status === "done";
  const errored = status?.status === "error";

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4"
    >
      <label className="text-sm text-zinc-300">
        MP4 video
        <input
          type="file"
          accept=".mp4,video/mp4"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-violet-600 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-violet-500"
        />
      </label>

      <div className="flex items-center gap-2">
        <button
          type="submit"
          disabled={submitting || !file || !styleId}
          className="rounded-md bg-violet-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "Uploading..." : "Stylize video"}
        </button>
        {jobId && (
          <span className="text-xs text-zinc-500">
            Job: <span className="font-mono">{jobId}</span>
          </span>
        )}
      </div>

      {status && !errored && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs text-zinc-400">
            <span className="capitalize">{status.status}</span>
            <span>
              {status.processed_frames}/{status.total_frames} frames ({percent}%)
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full bg-violet-500 transition-all duration-200"
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {done && jobId && (
        <a
          href={videoDownloadUrl(jobId)}
          className="inline-flex w-fit items-center rounded-md bg-emerald-500 px-3 py-1.5 text-sm font-medium text-zinc-950 hover:bg-emerald-400"
          download
        >
          Download stylized video
        </a>
      )}

      {(error || errored) && (
        <div className="rounded-md border border-red-900/50 bg-red-950/30 p-2 text-xs text-red-200">
          {error ?? status?.error ?? "Video processing failed"}
        </div>
      )}
    </form>
  );
}

export default VideoUpload;

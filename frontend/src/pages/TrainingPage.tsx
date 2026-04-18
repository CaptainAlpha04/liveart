import type { FormEvent } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  startTrainingFromSource,
  startTrainingFromUpload,
  stopTraining,
  trainingStatus as fetchTrainingStatus,
} from "../api/client";
import HyperparamForm, {
  DEFAULT_TRAINING_CONFIG,
} from "../components/HyperparamForm";
import LossChart from "../components/LossChart";
import ModelExport from "../components/ModelExport";
import StyleSourcePicker from "../components/StyleSourcePicker";
import TrainingLog from "../components/TrainingLog";
import { useTrainingSocket } from "../hooks/useTrainingSocket";
import type { StyleSource, TrainingConfig, TrainingStatus } from "../types";

type Mode = "predefined" | "upload";

function trainingSocketUrl(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/training`;
}

export function TrainingPage() {
  const [mode, setMode] = useState<Mode>("predefined");
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [styleImage, setStyleImage] = useState<File | null>(null);
  const [styleSource, setStyleSource] = useState<StyleSource | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const socketUrl = useMemo(trainingSocketUrl, []);
  const { events, latestEvent, connected, reset } =
    useTrainingSocket(socketUrl);

  // Poll /api/training/status every 2s so the UI reflects the current state
  // even if no events have been emitted recently (first few batches, etc.).
  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const s = await fetchTrainingStatus();
        if (!cancelled) setStatus(s);
      } catch {
        /* ignore transient */
      }
    };

    tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const isRunning = status?.state === "running";
  const isDone = status?.state === "done" || latestEvent?.status === "done";
  const trainedId =
    latestEvent?.model_id || status?.style_name || config.style_name;

  const handleSelectSource = useCallback(
    (source: StyleSource | null) => {
      setStyleSource(source);
      // Auto-fill style name from the chosen reference so the resulting model
      // gets a human-friendly id. User can still override it in the form.
      if (source) {
        setConfig((prev) => ({
          ...prev,
          style_name: source.name,
        }));
      }
    },
    []
  );

  const handleStart = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setError(null);

      if (mode === "upload" && !styleImage) {
        setError("Upload a style image first.");
        return;
      }
      if (mode === "predefined" && !styleSource) {
        setError("Pick a reference artwork first.");
        return;
      }

      setSubmitting(true);
      try {
        reset();
        if (mode === "upload" && styleImage) {
          await startTrainingFromUpload(styleImage, config);
        } else if (mode === "predefined" && styleSource) {
          await startTrainingFromSource(styleSource.id, config);
        }
        // Nudge status immediately; the socket will stream events from here.
        try {
          const s = await fetchTrainingStatus();
          setStatus(s);
        } catch {
          /* ignore */
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Start failed");
      } finally {
        setSubmitting(false);
      }
    },
    [config, mode, reset, styleImage, styleSource]
  );

  const handleStop = useCallback(async () => {
    setError(null);
    try {
      await stopTraining();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stop failed");
    }
  }, []);

  const progressPct = status
    ? Math.round((status.progress || 0) * 100)
    : latestEvent
    ? Math.round(
        ((latestEvent.batch || 0) / (latestEvent.total_batches || 1)) * 100
      )
    : 0;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 p-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Train a style</h1>
          <p className="text-xs text-zinc-400">
            Pick a curated reference artwork or upload your own, tune
            hyperparameters, and watch the loss curves converge in real time.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={[
              "inline-block h-2 w-2 rounded-full",
              connected ? "bg-emerald-400" : "bg-red-500",
            ].join(" ")}
          />
          <span className="text-zinc-400">
            {connected ? "Training socket connected" : "Disconnected"}
          </span>
        </div>
      </header>

      <form
        onSubmit={handleStart}
        className="flex flex-col gap-4 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4"
      >
        {/* Mode toggle */}
        <div
          role="tablist"
          aria-label="Training source"
          className="flex w-fit overflow-hidden rounded-md border border-zinc-700"
        >
          <button
            type="button"
            role="tab"
            aria-selected={mode === "predefined"}
            onClick={() => setMode("predefined")}
            disabled={isRunning}
            className={[
              "px-3 py-1.5 text-xs font-medium transition-colors",
              mode === "predefined"
                ? "bg-violet-600 text-white"
                : "bg-zinc-900 text-zinc-300 hover:bg-zinc-800",
              isRunning ? "cursor-not-allowed opacity-60" : "",
            ].join(" ")}
          >
            Predefined style
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "upload"}
            onClick={() => setMode("upload")}
            disabled={isRunning}
            className={[
              "px-3 py-1.5 text-xs font-medium transition-colors",
              mode === "upload"
                ? "bg-violet-600 text-white"
                : "bg-zinc-900 text-zinc-300 hover:bg-zinc-800",
              isRunning ? "cursor-not-allowed opacity-60" : "",
            ].join(" ")}
          >
            Upload custom image
          </button>
        </div>

        {/* Source input */}
        {mode === "predefined" ? (
          <div className="flex flex-col gap-2">
            <span className="text-sm text-zinc-300">Choose a reference artwork</span>
            <StyleSourcePicker
              selected={styleSource?.id ?? null}
              onSelect={handleSelectSource}
              disabled={isRunning}
            />
            {styleSource && (
              <p className="text-[11px] text-zinc-500">
                Selected: <span className="text-zinc-300">{styleSource.name}</span>
                {styleSource.artist && <> — {styleSource.artist}</>}
              </p>
            )}
          </div>
        ) : (
          <label className="text-sm">
            <span className="text-zinc-300">Style image</span>
            <input
              type="file"
              accept="image/*"
              disabled={isRunning}
              onChange={(e) => setStyleImage(e.target.files?.[0] ?? null)}
              className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-violet-600 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-violet-500 disabled:opacity-50"
            />
            {styleImage && (
              <p className="mt-1 text-[11px] text-zinc-500">
                {styleImage.name} ({Math.round(styleImage.size / 1024)} KB)
              </p>
            )}
          </label>
        )}

        <HyperparamForm
          value={config}
          onChange={setConfig}
          disabled={isRunning}
        />

        <div className="flex items-center gap-2">
          <button
            type="submit"
            disabled={submitting || isRunning}
            className="rounded-md bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Starting..." : "Start training"}
          </button>
          <button
            type="button"
            onClick={handleStop}
            disabled={!isRunning}
            className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Stop
          </button>

          {status && (
            <span className="ml-2 text-xs text-zinc-400">
              State: <span className="font-medium text-zinc-200">{status.state}</span>
              {isRunning && (
                <>
                  {" "}· epoch {status.epoch} · batch {status.batch}/{status.total_batches}{" "}
                  ({progressPct}%)
                </>
              )}
            </span>
          )}
        </div>

        {error && (
          <div className="rounded-md border border-red-900/50 bg-red-950/30 p-2 text-xs text-red-200">
            {error}
          </div>
        )}

        <p className="text-[11px] leading-relaxed text-zinc-500">
          Tip: training uses the images in{" "}
          <code className="rounded bg-zinc-800 px-1">data/coco_train/</code> as
          the content corpus. Set{" "}
          <code className="rounded bg-zinc-800 px-1">COCO_TRAIN_DIR</code> to
          point elsewhere. Each run takes roughly 5–15 min per style on a
          recent consumer GPU.
        </p>
      </form>

      <LossChart data={events} />

      <TrainingLog events={events} />

      {isDone && <ModelExport styleId={trainedId || config.style_name} />}
    </div>
  );
}

export default TrainingPage;

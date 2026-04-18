interface StatsBarProps {
  style: string;
  fps: number;
  inferenceMs: number;
  connected: boolean;
}

// Compact single-row stats bar used at the top of the inference page.
export function StatsBar({ style, fps, inferenceMs, connected }: StatsBarProps) {
  return (
    <div className="flex items-center justify-between gap-6 rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-2 text-sm">
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className={[
            "inline-block h-2.5 w-2.5 rounded-full",
            connected ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" : "bg-red-500",
          ].join(" ")}
        />
        <span className="text-zinc-300">
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      <div className="flex items-center gap-6 text-zinc-200">
        <div>
          <span className="text-zinc-500">Style:</span>{" "}
          <span className="font-medium">{style || "-"}</span>
        </div>
        <div>
          <span className="text-zinc-500">FPS:</span>{" "}
          <span className="font-mono tabular-nums">{fps.toFixed(1)}</span>
        </div>
        <div>
          <span className="text-zinc-500">Inference:</span>{" "}
          <span className="font-mono tabular-nums">
            {inferenceMs.toFixed(0)} ms
          </span>
        </div>
      </div>
    </div>
  );
}

export default StatsBar;

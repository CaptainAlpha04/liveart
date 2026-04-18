import { useEffect, useRef } from "react";
import type { TrainingEvent } from "../types";

interface TrainingLogProps {
  events: TrainingEvent[];
}

// Fixed-height scroll area that auto-scrolls to bottom whenever a new event
// arrives. Uses a sentinel ref at the bottom and scrollIntoView.
export function TrainingLog({ events }: TrainingLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length]);

  const fmt = (v: unknown, digits: number): string =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

  return (
    <div className="h-64 w-full overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs text-zinc-300">
      {events.length === 0 ? (
        <div className="text-zinc-600">Waiting for training events...</div>
      ) : (
        <ul className="flex flex-col gap-0.5">
          {events.map((e, i) => {
            const isTerminal =
              e.status === "done" ||
              e.status === "error" ||
              e.status === "stopped";
            if (isTerminal) {
              return (
                <li key={i} className="whitespace-nowrap">
                  <span
                    className={[
                      "rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide",
                      e.status === "done"
                        ? "bg-emerald-900/60 text-emerald-200"
                        : e.status === "error"
                        ? "bg-red-900/60 text-red-200"
                        : "bg-amber-900/60 text-amber-200",
                    ].join(" ")}
                  >
                    {e.status}
                  </span>
                  {e.model_id && (
                    <span className="ml-2 text-zinc-400">
                      model=<span className="text-zinc-200">{e.model_id}</span>
                    </span>
                  )}
                  {typeof e.elapsed_s === "number" && (
                    <span className="ml-2 text-zinc-500">
                      · elapsed {Math.round(e.elapsed_s)}s
                    </span>
                  )}
                </li>
              );
            }
            return (
              <li key={i} className="whitespace-nowrap">
                <span className="text-zinc-500">[e{e.epoch}</span>
                <span className="text-zinc-500"> b{e.batch}]</span>{" "}
                <span className="text-blue-300">c={fmt(e.content_loss, 3)}</span>{" "}
                <span className="text-pink-300">s={fmt(e.style_loss, 3)}</span>{" "}
                <span className="text-zinc-400">tv={fmt(e.tv_loss, 4)}</span>{" "}
                <span className="text-emerald-300">
                  total={fmt(e.total_loss, 3)}
                </span>
                {typeof e.eta_s === "number" && (
                  <span className="text-zinc-500">
                    {" "}
                    · eta {Math.round(e.eta_s)}s
                  </span>
                )}
              </li>
            );
          })}
          <div ref={bottomRef} />
        </ul>
      )}
    </div>
  );
}

export default TrainingLog;

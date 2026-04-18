import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchStyles, refreshModels } from "../api/client";
import StatsBar from "../components/StatsBar";
import StyleGrid from "../components/StyleGrid";
import StylizedCanvas from "../components/StylizedCanvas";
import VideoUpload from "../components/VideoUpload";
import WebcamFeed from "../components/WebcamFeed";
import { useInferenceSocket } from "../hooks/useInferenceSocket";
import type { StyleInfo } from "../types";

const FRAME_WIDTH = 480;
const FRAME_HEIGHT = 360;

// Build a same-origin ws:// (or wss://) URL that goes through the Vite proxy.
function inferenceSocketUrl(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/inference`;
}

type SourceMode = "webcam" | "video";

export function InferencePage() {
  const [styles, setStyles] = useState<StyleInfo[]>([]);
  const [stylesError, setStylesError] = useState<string | null>(null);
  const [activeStyle, setActiveStyle] = useState<string>("");
  const [sourceMode, setSourceMode] = useState<SourceMode>("webcam");
  const [refreshing, setRefreshing] = useState(false);

  const socketUrl = useMemo(inferenceSocketUrl, []);
  const { sendFrame, lastFrame, stats, connected, error } =
    useInferenceSocket(socketUrl);

  const loadStyles = useCallback(
    async (useRefresh = false): Promise<void> => {
      try {
        const list = useRefresh ? await refreshModels() : await fetchStyles();
        setStyles(list);
        setStylesError(null);
        setActiveStyle((prev) => (prev && list.some((s) => s.id === prev) ? prev : list[0]?.id ?? ""));
      } catch (err: unknown) {
        setStylesError(err instanceof Error ? err.message : "Load failed");
      }
    },
    []
  );

  // Initial load + reload whenever the tab regains focus so newly-trained
  // models show up without a hard refresh.
  useEffect(() => {
    void loadStyles(false);
    const onFocus = () => void loadStyles(false);
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [loadStyles]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadStyles(true);
    } finally {
      setRefreshing(false);
    }
  }, [loadStyles]);

  const handleFrame = useCallback(
    (base64: string) => {
      if (!activeStyle) return;
      sendFrame(activeStyle, base64);
    },
    [activeStyle, sendFrame]
  );

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 p-4">
      <StatsBar
        style={stats.style || activeStyle}
        fps={stats.fps}
        inferenceMs={stats.inferenceMs}
        connected={connected}
      />

      {error && (
        <div className="rounded-md border border-red-900/50 bg-red-950/30 p-2 text-xs text-red-200">
          {error}
        </div>
      )}

      <div className="flex items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-zinc-800">
          {(["webcam", "video"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setSourceMode(m)}
              className={[
                "px-3 py-1.5 text-xs font-medium capitalize transition",
                sourceMode === m
                  ? "bg-violet-600 text-white"
                  : "bg-zinc-900 text-zinc-300 hover:bg-zinc-800",
              ].join(" ")}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_auto]">
        <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <StylizedCanvas
            frame={lastFrame}
            width={FRAME_WIDTH}
            height={FRAME_HEIGHT}
          />
        </div>

        <div className="flex w-full flex-col gap-3 lg:w-80">
          {sourceMode === "webcam" ? (
            <WebcamFeed
              onFrame={handleFrame}
              width={FRAME_WIDTH}
              height={FRAME_HEIGHT}
              active={Boolean(activeStyle)}
              intervalMs={50}
            />
          ) : (
            <VideoUpload styleId={activeStyle} />
          )}
        </div>
      </div>

      <section className="rounded-lg border border-zinc-800 bg-zinc-900/40">
        <div className="flex items-center justify-between px-4 py-2 text-xs uppercase tracking-wide text-zinc-400">
          <div className="flex items-center gap-3">
            <span>Styles ({styles.length})</span>
            {stylesError && (
              <span className="normal-case text-red-400">Error: {stylesError}</span>
            )}
          </div>
          <button
            type="button"
            onClick={handleRefresh}
            disabled={refreshing}
            className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            title="Rescan backend/models/ — pick up newly-trained styles"
          >
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
        <StyleGrid
          styles={styles}
          selected={activeStyle}
          onSelect={setActiveStyle}
        />
      </section>
    </div>
  );
}

export default InferencePage;

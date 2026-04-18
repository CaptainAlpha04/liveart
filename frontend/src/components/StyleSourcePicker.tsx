import { useEffect, useState } from "react";
import { fetchStyleSources } from "../api/client";
import type { StyleSource } from "../types";

interface Props {
  selected: string | null;
  onSelect: (source: StyleSource | null) => void;
  disabled?: boolean;
}

/**
 * Grid of curated reference artworks from ``backend/style_sources/``. Selecting
 * one lets the user train without uploading their own image.
 */
export function StyleSourcePicker({ selected, onSelect, disabled }: Props) {
  const [sources, setSources] = useState<StyleSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchStyleSources()
      .then((data) => {
        if (cancelled) return;
        setSources(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load sources");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="rounded-md border border-zinc-800 bg-zinc-900/50 p-3 text-xs text-zinc-400">
        Loading reference artworks...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-red-900/50 bg-red-950/30 p-3 text-xs text-red-200">
        {error}
      </div>
    );
  }

  if (sources.length === 0) {
    return (
      <div className="rounded-md border border-zinc-800 bg-zinc-900/50 p-3 text-xs text-zinc-400">
        No reference artworks found. Run{" "}
        <code className="rounded bg-zinc-800 px-1">
          python scripts/download_style_images.py
        </code>{" "}
        to fetch the curated set, or switch to "Upload custom image".
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
      {sources.map((s) => {
        const isSelected = selected === s.id;
        return (
          <button
            key={s.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(isSelected ? null : s)}
            className={[
              "group relative overflow-hidden rounded-lg border text-left transition-all",
              isSelected
                ? "border-violet-500 ring-2 ring-violet-500/60"
                : "border-zinc-800 hover:border-zinc-600",
              disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
            ].join(" ")}
            aria-pressed={isSelected}
          >
            <div className="aspect-[4/3] w-full bg-zinc-900">
              <img
                src={s.image_url}
                alt={s.name}
                loading="lazy"
                className="h-full w-full object-cover"
              />
            </div>
            <div className="flex flex-col gap-0.5 px-2 py-1.5">
              <span className="truncate text-xs font-medium text-zinc-100">
                {s.name}
              </span>
              {s.artist && (
                <span className="truncate text-[10px] text-zinc-500">
                  {s.artist}
                </span>
              )}
            </div>
            {isSelected && (
              <div className="absolute right-1.5 top-1.5 rounded-full bg-violet-500 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                ✓
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default StyleSourcePicker;

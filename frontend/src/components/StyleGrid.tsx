import { memo } from "react";
import type { StyleInfo } from "../types";

interface StyleGridProps {
  styles: StyleInfo[];
  selected: string;
  onSelect: (id: string) => void;
}

// Horizontal scroll strip of style cards. The selected card gets a violet
// ring; cards are small (thumb + label) so many fit on a desktop width.
function StyleGridImpl({ styles, selected, onSelect }: StyleGridProps) {
  return (
    <div className="no-scrollbar w-full overflow-x-auto">
      <div className="flex gap-3 px-4 py-3">
        {styles.map((s) => {
          const isSelected = s.id === selected;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => onSelect(s.id)}
              className={[
                "group relative flex w-28 flex-shrink-0 flex-col items-center gap-2",
                "rounded-lg border bg-zinc-900/50 p-2 text-left transition",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400",
                isSelected
                  ? "border-violet-500 ring-2 ring-violet-500/70"
                  : "border-zinc-800 hover:border-zinc-600",
              ].join(" ")}
              aria-pressed={isSelected}
            >
              <div className="h-20 w-full overflow-hidden rounded-md bg-zinc-800">
                <img
                  src={s.thumbnail_url}
                  alt={s.name}
                  loading="lazy"
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.visibility =
                      "hidden";
                  }}
                />
              </div>
              <div className="w-full">
                <div className="truncate text-xs font-medium text-zinc-100">
                  {s.name}
                </div>
                {s.artist && (
                  <div className="truncate text-[10px] text-zinc-400">
                    {s.artist}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export const StyleGrid = memo(StyleGridImpl);
export default StyleGrid;

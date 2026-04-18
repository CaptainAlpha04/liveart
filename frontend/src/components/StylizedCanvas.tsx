import { useEffect, useRef } from "react";

interface StylizedCanvasProps {
  frame: string | null;
  width: number;
  height: number;
  className?: string;
}

// Draws incoming base64 JPEG strings onto a <canvas>. We create a new Image()
// per frame, wait for onload, then draw. For smooth painting we do not clear
// before drawImage — drawImage with identical dims overwrites in place.
export function StylizedCanvas({
  frame,
  width,
  height,
  className,
}: StylizedCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!frame) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, width, height);
    };
    img.src = `data:image/jpeg;base64,${frame}`;
  }, [frame, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={
        className ??
        "rounded-lg border border-zinc-800 bg-zinc-900 shadow-lg shadow-black/40"
      }
    />
  );
}

export default StylizedCanvas;

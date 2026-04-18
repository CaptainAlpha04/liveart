import { useEffect } from "react";
import { useWebcam } from "../hooks/useWebcam";

interface WebcamFeedProps {
  onFrame: (base64: string) => void;
  width: number;
  height: number;
  active: boolean;
  // Interval between captures in ms. 50ms ≈ 20fps matches the inference cadence.
  intervalMs?: number;
  showPreview?: boolean;
}

// Renders a <video> element bound to the webcam and runs a setInterval loop
// that captures a JPEG frame every `intervalMs` and forwards the base64 to
// the parent via onFrame. The setInterval is torn down on unmount / toggle.
export function WebcamFeed({
  onFrame,
  width,
  height,
  active,
  intervalMs = 50,
  showPreview = true,
}: WebcamFeedProps) {
  const { videoRef, canvasRef, captureFrame, error, ready } = useWebcam(
    width,
    height
  );

  useEffect(() => {
    if (!active || !ready) return;
    const id = window.setInterval(() => {
      const b64 = captureFrame(0.8);
      if (b64) onFrame(b64);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [active, ready, captureFrame, onFrame, intervalMs]);

  if (error) {
    return (
      <div className="rounded-md border border-red-900/50 bg-red-950/30 p-3 text-sm text-red-200">
        Webcam error: {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={
          showPreview
            ? "h-28 w-40 rounded-md border border-zinc-800 object-cover opacity-80"
            : "hidden"
        }
      />
      {/* Hidden capture canvas used by useWebcam.captureFrame */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

export default WebcamFeed;

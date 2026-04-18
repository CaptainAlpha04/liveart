import type { RefObject } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

// useWebcam wraps getUserMedia. It owns two refs: `videoRef` must be attached
// to a <video> element so the browser can paint decoded frames, and
// `canvasRef` may optionally be attached to a hidden <canvas> (the hook
// creates an internal canvas if none is wired up).
//
// captureFrame() returns a base64-encoded JPEG string *without* the
// `data:image/jpeg;base64,` prefix — the backend inference WS expects the
// raw base64 payload.
export interface UseWebcamResult {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  captureFrame: (quality?: number) => string | null;
  error: string | null;
  ready: boolean;
}

export function useWebcam(width: number, height: number): UseWebcamResult {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Fallback canvas when the consumer does not render a <canvas> element —
  // captureFrame still needs somewhere to draw.
  const internalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices?.getUserMedia
      ) {
        setError("Webcam not supported in this browser");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width, height },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play().catch(() => {
              /* autoplay may be blocked; caller can retry */
            });
            setReady(true);
          };
        }
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to access webcam";
        setError(msg);
      }
    }

    start();

    return () => {
      cancelled = true;
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      streamRef.current = null;
      setReady(false);
    };
  }, [width, height]);

  const captureFrame = useCallback(
    (quality = 0.8): string | null => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return null;

      let canvas = canvasRef.current;
      if (!canvas) {
        if (!internalCanvasRef.current) {
          internalCanvasRef.current = document.createElement("canvas");
        }
        canvas = internalCanvasRef.current;
      }
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, width, height);
      const dataUrl = canvas.toDataURL("image/jpeg", quality);
      // Strip the `data:image/jpeg;base64,` prefix — backend wants raw b64.
      const idx = dataUrl.indexOf(",");
      return idx >= 0 ? dataUrl.slice(idx + 1) : dataUrl;
    },
    [width, height]
  );

  return { videoRef, canvasRef, captureFrame, error, ready };
}

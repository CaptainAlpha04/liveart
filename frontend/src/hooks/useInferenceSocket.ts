import { useCallback, useEffect, useRef, useState } from "react";
import type { InferenceStats } from "../types";

// Backoff schedule in milliseconds. Matches the spec: 1s, 2s, 4s, 8s, 16s
// capped at 5 attempts.
const BACKOFFS = [1000, 2000, 4000, 8000, 16000];

export interface UseInferenceSocketResult {
  sendFrame: (style: string, base64: string) => void;
  lastFrame: string | null;
  stats: InferenceStats;
  connected: boolean;
  error: string | null;
}

interface ServerFrameMessage {
  frame: string;
  inference_ms: number;
  fps: number;
  style: string;
}

interface ServerErrorMessage {
  error: string;
  detail?: string;
}

type ServerMessage = ServerFrameMessage | ServerErrorMessage;

function isFrameMessage(msg: ServerMessage): msg is ServerFrameMessage {
  return typeof (msg as ServerFrameMessage).frame === "string";
}

// useInferenceSocket owns a single WebSocket connection to /ws/inference with
// exponential-backoff reconnection. Consumers call `sendFrame` each capture
// tick; the hook exposes the most recent stylized frame plus running stats.
export function useInferenceSocket(url: string): UseInferenceSocketResult {
  const socketRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const [lastFrame, setLastFrame] = useState<string | null>(null);
  const [stats, setStats] = useState<InferenceStats>({
    fps: 0,
    inferenceMs: 0,
    style: "",
  });
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (attemptsRef.current >= BACKOFFS.length + 1) {
      setError("Inference socket: max reconnect attempts reached");
      return;
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "WebSocket failed");
      return;
    }
    socketRef.current = ws;

    ws.onopen = () => {
      attemptsRef.current = 0;
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as ServerMessage;
        if (isFrameMessage(msg)) {
          setLastFrame(msg.frame);
          setStats({
            fps: msg.fps,
            inferenceMs: msg.inference_ms,
            style: msg.style,
          });
        } else if ("error" in msg) {
          setError(msg.detail ? `${msg.error}: ${msg.detail}` : msg.error);
        }
      } catch {
        // Ignore non-JSON frames; the server only sends JSON.
      }
    };

    ws.onerror = () => {
      setError("Inference socket error");
    };

    ws.onclose = () => {
      setConnected(false);
      socketRef.current = null;
      if (!mountedRef.current) return;
      const idx = attemptsRef.current;
      if (idx < BACKOFFS.length) {
        const delay = BACKOFFS[idx];
        attemptsRef.current = idx + 1;
        reconnectTimerRef.current = setTimeout(connect, delay);
      } else {
        setError("Inference socket: max reconnect attempts reached");
      }
    };
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    attemptsRef.current = 0;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      const ws = socketRef.current;
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      socketRef.current = null;
    };
  }, [connect]);

  const sendFrame = useCallback((style: string, base64: string) => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        style,
        frame: base64,
      })
    );
  }, []);

  return { sendFrame, lastFrame, stats, connected, error };
}

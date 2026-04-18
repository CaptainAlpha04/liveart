import { useCallback, useEffect, useRef, useState } from "react";
import type { TrainingEvent } from "../types";

export interface UseTrainingSocketResult {
  events: TrainingEvent[];
  latestEvent: TrainingEvent | null;
  connected: boolean;
  reset: () => void;
}

// useTrainingSocket opens /ws/training and accumulates every event pushed by
// the backend into a flat array. The training page feeds this array into the
// Recharts loss chart and the log list.
export function useTrainingSocket(url: string): UseTrainingSocketResult {
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      return;
    }
    socketRef.current = ws;

    ws.onopen = () => {
      if (mountedRef.current) setConnected(true);
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as TrainingEvent;
        if (mountedRef.current) setEvents((prev) => [...prev, msg]);
      } catch {
        /* swallow malformed events */
      }
    };
    ws.onclose = () => {
      if (mountedRef.current) setConnected(false);
    };
    ws.onerror = () => {
      if (mountedRef.current) setConnected(false);
    };

    return () => {
      mountedRef.current = false;
      ws.onclose = null;
      ws.close();
      socketRef.current = null;
    };
  }, [url]);

  const latestEvent = events.length ? events[events.length - 1] : null;

  const reset = useCallback(() => {
    setEvents([]);
  }, []);

  return { events, latestEvent, connected, reset };
}

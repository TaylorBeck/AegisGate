"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { TelemetryEventSchema, type TelemetryEvent } from "@/lib/contracts";
import {
  initialTelemetryState,
  telemetryReducer,
} from "@/lib/telemetry-store";

export function useTelemetry(gatewayUrl?: string) {
  const [state, dispatch] = useReducer(telemetryReducer, initialTelemetryState);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const bufferedEventsRef = useRef<TelemetryEvent[]>([]);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    const baseUrl =
      gatewayUrl ||
      process.env.NEXT_PUBLIC_GATEWAY_URL ||
      "http://localhost:8000";
    const eventsUrl = `${baseUrl.replace(/\/$/, "")}/v1/aegis/events`;

    let eventSource: EventSource | null = null;
    let isCleanedUp = false;

    function flush() {
      if (bufferedEventsRef.current.length > 0) {
        const eventsToFlush = [...bufferedEventsRef.current];
        bufferedEventsRef.current = [];
        dispatch({ type: "ADD_EVENTS", events: eventsToFlush });
      }
      rafIdRef.current = null;
    }

    function queueEvent(event: TelemetryEvent) {
      bufferedEventsRef.current.push(event);
      if (rafIdRef.current === null) {
        rafIdRef.current = requestAnimationFrame(flush);
      }
    }

    try {
      eventSource = new EventSource(eventsUrl);

      eventSource.onopen = () => {
        if (!isCleanedUp) {
          setIsConnected(true);
          setConnectionError(null);
        }
      };

      eventSource.onmessage = (e: MessageEvent) => {
        if (isCleanedUp) return;
        try {
          if (!e.data || e.data === ": ping") return;
          const json = JSON.parse(e.data as string) as unknown;
          const parsed = TelemetryEventSchema.safeParse(json);
          if (parsed.success) {
            queueEvent(parsed.data);
          } else {
            console.warn("Failed to validate telemetry event:", parsed.error);
          }
        } catch (err) {
          console.warn("Failed to parse SSE JSON:", err);
        }
      };

      eventSource.onerror = () => {
        if (!isCleanedUp) {
          setIsConnected(false);
          setConnectionError("Connecting to AegisGate telemetry stream...");
        }
      };
    } catch (err) {
      setTimeout(() => {
        if (!isCleanedUp) {
          setConnectionError(err instanceof Error ? err.message : String(err));
        }
      }, 0);
    }

    return () => {
      isCleanedUp = true;
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
      flush();
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [gatewayUrl]);

  return { state, dispatch, isConnected, connectionError };
}

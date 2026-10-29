import { describe, expect, it } from "vitest";
import type { RequestCompletedEvent, TelemetryEvent } from "./contracts";
import {
  computeKPIs,
  computePercentile,
  initialTelemetryState,
  MAX_EVENTS,
  telemetryReducer,
} from "./telemetry-store";

function makeCompletedEvent(
  id: string,
  status: "HIT_L1" | "HIT_L2" | "MISS",
  overheadMs: number,
  savedUsd: number,
  actualUsd: number,
  ttftMs: number | null = 10,
): RequestCompletedEvent {
  return {
    type: "request.completed",
    request_id: `req-${id}`,
    ts: new Date().toISOString(),
    tenant_id: "sandbox",
    prompt_preview: "test prompt",
    total_ms: overheadMs + 50,
    status: "ok",
    finops: {
      request_id: `req-${id}`,
      cache_status: status,
      similarity_score: status === "HIT_L2" ? 0.95 : null,
      routed_tier: "SIMPLE",
      routed_model: "test-model",
      latency_breakdown_ms: {
        guardrail: 0.1,
        l1_lookup: 0.5,
        onnx_embed: 4.0,
        l2_hnsw: 1.5,
        router: 0.2,
      },
      cost_actual_usd: actualUsd,
      cost_saved_usd: savedUsd,
      gateway_overhead_ms: overheadMs,
      ttft_ms: ttftMs,
      tokens_in: 10,
      tokens_out: 20,
    },
  };
}

describe("telemetry-store", () => {
  describe("computePercentile", () => {
    it("returns 0 for empty arrays", () => {
      expect(computePercentile([], 95)).toBe(0);
    });

    it("computes exact p95 for a uniform distribution 0..100", () => {
      const numbers = Array.from({ length: 101 }, (_, i) => i);
      expect(computePercentile(numbers, 95)).toBe(95);
    });

    it("handles single-element array", () => {
      expect(computePercentile([42], 95)).toBe(42);
    });
  });

  describe("computeKPIs", () => {
    it("calculates correct hit ratios, savings, and p95 overhead", () => {
      const events: TelemetryEvent[] = [
        makeCompletedEvent("1", "HIT_L1", 1.2, 0.01, 0.0, null),
        makeCompletedEvent("2", "HIT_L2", 7.5, 0.01, 0.0, 7.5),
        makeCompletedEvent("3", "MISS", 8.0, 0.005, 0.002, 12.0),
        makeCompletedEvent("4", "MISS", 10.0, 0.0, 0.003, 15.0),
      ];

      const kpis = computeKPIs(events);
      expect(kpis.totalRequests).toBe(4);
      expect(kpis.hitRatioL1).toBe(0.25);
      expect(kpis.hitRatioL2).toBe(0.25);
      expect(kpis.combinedHitRatio).toBe(0.5);
      expect(kpis.totalSavedUsd).toBeCloseTo(0.025, 5);
      expect(kpis.totalActualUsd).toBeCloseTo(0.005, 5);
      // ttft_ms values: 7.5, 12.0, 15.0 -> avg = 34.5 / 3 = 11.5
      expect(kpis.avgUpstreamTtftMs).toBeCloseTo(11.5, 2);
    });
  });

  describe("telemetryReducer ring buffer", () => {
    it("enforces MAX_EVENTS cap and preserves newest events", () => {
      let state = initialTelemetryState;
      const batch1: TelemetryEvent[] = Array.from({ length: 300 }, (_, i) =>
        makeCompletedEvent(`batch1-${i}`, "MISS", 8.0, 0.001, 0.001),
      );
      state = telemetryReducer(state, { type: "ADD_EVENTS", events: batch1 });
      expect(state.events.length).toBe(300);

      const batch2: TelemetryEvent[] = Array.from({ length: 300 }, (_, i) =>
        makeCompletedEvent(`batch2-${i}`, "HIT_L1", 1.0, 0.01, 0.0),
      );
      state = telemetryReducer(state, { type: "ADD_EVENTS", events: batch2 });
      expect(state.events.length).toBe(MAX_EVENTS);
      // Newest event in batch2 should be at the front
      expect(state.events[0].type).toBe("request.completed");
      if (state.events[0].type === "request.completed") {
        expect(state.events[0].request_id).toBe("req-batch2-0");
      }
    });

    it("tracks circuit breaker state transitions", () => {
      let state = initialTelemetryState;
      state = telemetryReducer(state, {
        type: "ADD_EVENT",
        event: {
          type: "breaker.state",
          provider: "mock-primary",
          state: "OPEN",
          ts: new Date().toISOString(),
        },
      });
      expect(state.breakers["mock-primary"]).toBe("OPEN");

      state = telemetryReducer(state, {
        type: "ADD_EVENT",
        event: {
          type: "breaker.state",
          provider: "mock-primary",
          state: "HALF_OPEN",
          ts: new Date().toISOString(),
        },
      });
      expect(state.breakers["mock-primary"]).toBe("HALF_OPEN");
    });
  });
});

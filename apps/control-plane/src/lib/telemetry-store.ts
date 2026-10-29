import {
  type BreakerName,
  type RequestCompletedEvent,
  type TelemetryEvent,
} from "./contracts";

export const MAX_EVENTS = 500;

export interface TelemetryKPIs {
  totalRequests: number;
  totalSavedUsd: number;
  totalActualUsd: number;
  hitRatioL1: number;
  hitRatioL2: number;
  combinedHitRatio: number;
  p95GatewayOverheadMs: number;
  avgUpstreamTtftMs: number;
}

export interface TelemetryState {
  events: TelemetryEvent[];
  breakers: Record<string, BreakerName>;
  activeRuns: Record<string, { scenario: string; count: number; completed: boolean }>;
  kpis: TelemetryKPIs;
}

export function computePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function computeKPIs(events: TelemetryEvent[]): TelemetryKPIs {
  const completedRequests: RequestCompletedEvent[] = [];
  for (const event of events) {
    if (event.type === "request.completed") {
      completedRequests.push(event);
    }
  }

  const totalRequests = completedRequests.length;
  if (totalRequests === 0) {
    return {
      totalRequests: 0,
      totalSavedUsd: 0,
      totalActualUsd: 0,
      hitRatioL1: 0,
      hitRatioL2: 0,
      combinedHitRatio: 0,
      p95GatewayOverheadMs: 0,
      avgUpstreamTtftMs: 0,
    };
  }

  let l1Hits = 0;
  let l2Hits = 0;
  let totalSavedUsd = 0;
  let totalActualUsd = 0;
  const overheads: number[] = [];
  const ttfts: number[] = [];

  for (const req of completedRequests) {
    const finops = req.finops;
    if (finops.cache_status === "HIT_L1") l1Hits++;
    else if (finops.cache_status === "HIT_L2") l2Hits++;

    totalSavedUsd += finops.cost_saved_usd;
    totalActualUsd += finops.cost_actual_usd;

    overheads.push(finops.gateway_overhead_ms);
    if (finops.ttft_ms !== null && finops.ttft_ms !== undefined) {
      ttfts.push(finops.ttft_ms);
    }
  }

  const hitRatioL1 = l1Hits / totalRequests;
  const hitRatioL2 = l2Hits / totalRequests;
  const combinedHitRatio = (l1Hits + l2Hits) / totalRequests;
  const p95GatewayOverheadMs = computePercentile(overheads, 95);
  const avgUpstreamTtftMs =
    ttfts.length > 0 ? ttfts.reduce((sum, v) => sum + v, 0) / ttfts.length : 0;

  return {
    totalRequests,
    totalSavedUsd,
    totalActualUsd,
    hitRatioL1,
    hitRatioL2,
    combinedHitRatio,
    p95GatewayOverheadMs,
    avgUpstreamTtftMs,
  };
}

export const initialTelemetryState: TelemetryState = {
  events: [],
  breakers: {},
  activeRuns: {},
  kpis: {
    totalRequests: 0,
    totalSavedUsd: 0,
    totalActualUsd: 0,
    hitRatioL1: 0,
    hitRatioL2: 0,
    combinedHitRatio: 0,
    p95GatewayOverheadMs: 0,
    avgUpstreamTtftMs: 0,
  },
};

export type TelemetryAction =
  | { type: "ADD_EVENT"; event: TelemetryEvent }
  | { type: "ADD_EVENTS"; events: TelemetryEvent[] }
  | { type: "CLEAR" };

export function telemetryReducer(
  state: TelemetryState,
  action: TelemetryAction,
): TelemetryState {
  switch (action.type) {
    case "CLEAR":
      return initialTelemetryState;

    case "ADD_EVENT":
      return telemetryReducer(state, {
        type: "ADD_EVENTS",
        events: [action.event],
      });

    case "ADD_EVENTS": {
      if (action.events.length === 0) return state;

      // Prepend newest events first, keep max MAX_EVENTS
      const newEvents = [...action.events, ...state.events].slice(0, MAX_EVENTS);

      const nextBreakers = { ...state.breakers };
      const nextRuns = { ...state.activeRuns };

      for (const event of action.events) {
        if (event.type === "breaker.state") {
          nextBreakers[event.provider] = event.state;
        } else if (event.type === "simulation.started") {
          nextRuns[event.run_id] = {
            scenario: event.scenario,
            count: event.count,
            completed: false,
          };
        } else if (event.type === "simulation.completed") {
          if (nextRuns[event.run_id]) {
            nextRuns[event.run_id] = {
              ...nextRuns[event.run_id],
              completed: true,
            };
          }
        }
      }

      return {
        events: newEvents,
        breakers: nextBreakers,
        activeRuns: nextRuns,
        kpis: computeKPIs(newEvents),
      };
    }

    default:
      return state;
  }
}


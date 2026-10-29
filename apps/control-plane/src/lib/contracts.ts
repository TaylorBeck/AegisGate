import { z } from "zod";

export const CacheStatusSchema = z.enum(["HIT_L1", "HIT_L2", "MISS"]);
export type CacheStatus = z.infer<typeof CacheStatusSchema>;

export const TierSchema = z.enum(["SIMPLE", "REASONING", "AGENTIC_CODE"]);
export type Tier = z.infer<typeof TierSchema>;

export const LatencyBreakdownSchema = z.object({
  guardrail: z.number(),
  l1_lookup: z.number(),
  onnx_embed: z.number().nullable().optional(),
  l2_hnsw: z.number().nullable().optional(),
  router: z.number().nullable().optional(),
});
export type LatencyBreakdown = z.infer<typeof LatencyBreakdownSchema>;

export const FailoverInfoSchema = z.object({
  from_provider: z.string(),
  reason: z.string(),
});
export type FailoverInfo = z.infer<typeof FailoverInfoSchema>;

export const XAegisFinOpsSchema = z.object({
  cache_status: CacheStatusSchema,
  similarity_score: z.number().nullable(),
  routed_tier: TierSchema,
  routed_model: z.string(),
  latency_breakdown_ms: LatencyBreakdownSchema,
  cost_actual_usd: z.number(),
  cost_saved_usd: z.number(),
  request_id: z.string().default(""),
  provider: z.string().nullable().optional(),
  failover: FailoverInfoSchema.nullable().optional(),
  gateway_overhead_ms: z.number().default(0),
  ttft_ms: z.number().nullable().optional(),
  tokens_in: z.number().default(0),
  tokens_out: z.number().default(0),
});
export type XAegisFinOps = z.infer<typeof XAegisFinOpsSchema>;

export const BreakerNameSchema = z.enum(["CLOSED", "OPEN", "HALF_OPEN"]);
export type BreakerName = z.infer<typeof BreakerNameSchema>;

export const RequestCompletedEventSchema = z.object({
  type: z.literal("request.completed"),
  request_id: z.string(),
  ts: z.string(),
  tenant_id: z.string(),
  run_id: z.string().nullable().optional(),
  scenario: z.string().nullable().optional(),
  prompt_preview: z.string(),
  finops: XAegisFinOpsSchema,
  total_ms: z.number(),
  status: z.string(),
});
export type RequestCompletedEvent = z.infer<typeof RequestCompletedEventSchema>;

export const BreakerStateEventSchema = z.object({
  type: z.literal("breaker.state"),
  provider: z.string(),
  state: BreakerNameSchema,
  ts: z.string(),
});
export type BreakerStateEvent = z.infer<typeof BreakerStateEventSchema>;

export const SimulationStartedEventSchema = z.object({
  type: z.literal("simulation.started"),
  scenario: z.string(),
  run_id: z.string(),
  count: z.number(),
});
export type SimulationStartedEvent = z.infer<typeof SimulationStartedEventSchema>;

export const SimulationCompletedEventSchema = z.object({
  type: z.literal("simulation.completed"),
  scenario: z.string(),
  run_id: z.string(),
  count: z.number(),
});
export type SimulationCompletedEvent = z.infer<typeof SimulationCompletedEventSchema>;

export const TelemetryEventSchema = z.discriminatedUnion("type", [
  RequestCompletedEventSchema,
  BreakerStateEventSchema,
  SimulationStartedEventSchema,
  SimulationCompletedEventSchema,
]);
export type TelemetryEvent = z.infer<typeof TelemetryEventSchema>;


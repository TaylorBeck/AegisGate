"use client";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { LatencyWaterfall } from "./latency-waterfall";
import { SimilarityGauge } from "./similarity-gauge";
import { useTelemetryContext } from "./telemetry-provider";

export function TraceDrawer() {
  const { selectedRequest, setSelectedRequest } = useTelemetryContext();
  const isOpen = selectedRequest !== null;

  if (!selectedRequest) {
    return null;
  }

  const { finops, request_id, ts, tenant_id, scenario, run_id, prompt_preview, total_ms } =
    selectedRequest;

  return (
    <Sheet open={isOpen} onOpenChange={(open: boolean) => !open && setSelectedRequest(null)}>
      <SheetContent
        side="right"
        className="w-full sm:max-w-xl border-l border-border/40 bg-card p-0 text-foreground overflow-hidden flex flex-col font-sans"
      >
        <SheetHeader className="border-b border-border/40 p-4">
          <div className="flex items-center gap-2">
            {finops.cache_status === "HIT_L1" && (
              <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30 text-[10px] font-mono font-semibold px-2 py-0.5">
                HIT_L1 (EXACT)
              </Badge>
            )}
            {finops.cache_status === "HIT_L2" && (
              <Badge className="bg-cyan-500/15 text-cyan-400 border-cyan-500/30 text-[10px] font-mono font-semibold px-2 py-0.5">
                HIT_L2 (SEMANTIC)
              </Badge>
            )}
            {finops.cache_status === "MISS" && (
              <Badge variant="outline" className="border-border/60 bg-muted/40 text-muted-foreground text-[10px] font-mono px-2 py-0.5">
                CACHE MISS
              </Badge>
            )}
            <Badge variant="secondary" className="text-[10px] font-mono">
              {finops.routed_tier}
            </Badge>
          </div>
          <SheetTitle className="text-sm font-mono font-semibold truncate pt-1">
            {request_id}
          </SheetTitle>
          <SheetDescription className="text-xs font-mono text-muted-foreground">
            {new Date(ts).toLocaleString()} · Tenant: {tenant_id}
            {scenario && ` · Scenario: ${scenario}`}
            {run_id && ` · Run: ${run_id}`}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1 p-4 space-y-5">
          <div className="space-y-4">
            {/* Prompt preview */}
            <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 space-y-1">
              <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider font-sans">
                Scrubbed Prompt Preview
              </span>
              <p className="text-xs text-foreground/90 font-sans leading-relaxed whitespace-pre-wrap">
                {prompt_preview}
              </p>
            </div>

            {/* FinOps Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="rounded border border-border/40 bg-card/60 p-2">
                <span className="text-[10px] text-muted-foreground uppercase font-sans">Actual Cost</span>
                <div className="text-xs font-mono font-bold text-foreground tabular-nums">
                  ${finops.cost_actual_usd.toFixed(6)}
                </div>
              </div>
              <div className="rounded border border-border/40 bg-card/60 p-2">
                <span className="text-[10px] text-muted-foreground uppercase font-sans">Cost Saved</span>
                <div className="text-xs font-mono font-bold text-emerald-400 tabular-nums">
                  ${finops.cost_saved_usd.toFixed(6)}
                </div>
              </div>
              <div className="rounded border border-border/40 bg-card/60 p-2">
                <span className="text-[10px] text-muted-foreground uppercase font-sans">Tokens In / Out</span>
                <div className="text-xs font-mono font-bold text-foreground tabular-nums">
                  {finops.tokens_in} / {finops.tokens_out}
                </div>
              </div>
              <div className="rounded border border-border/40 bg-card/60 p-2">
                <span className="text-[10px] text-muted-foreground uppercase font-sans">Total Latency</span>
                <div className="text-xs font-mono font-bold text-foreground tabular-nums">
                  {total_ms.toFixed(1)}ms
                </div>
              </div>
            </div>

            {/* Model & Provider Info */}
            <div className="rounded-lg border border-border/40 bg-card/40 p-3 text-xs space-y-2 font-sans">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Routed Model:</span>
                <span className="font-mono font-semibold text-foreground">{finops.routed_model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Provider:</span>
                <span className="font-mono font-semibold text-foreground">{finops.provider ?? "N/A"}</span>
              </div>
              {finops.ttft_ms !== null && finops.ttft_ms !== undefined && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Time to First Token (TTFT):</span>
                  <span className="font-mono font-semibold text-foreground tabular-nums">{finops.ttft_ms.toFixed(1)}ms</span>
                </div>
              )}
              {finops.failover && (
                <div className="rounded border border-rose-500/30 bg-rose-500/10 p-2 text-[11px] text-rose-300">
                  <strong>Failover Active:</strong> Failed from {finops.failover.from_provider} ({finops.failover.reason})
                </div>
              )}
            </div>

            {/* Semantic Gauge */}
            <div>
              <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider font-sans block mb-1.5">
                L2 Semantic Cache Evaluation
              </span>
              <SimilarityGauge score={finops.similarity_score} />
            </div>

            {/* Latency Waterfall */}
            <div>
              <LatencyWaterfall
                breakdown={finops.latency_breakdown_ms}
                gatewayOverheadMs={finops.gateway_overhead_ms}
              />
            </div>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

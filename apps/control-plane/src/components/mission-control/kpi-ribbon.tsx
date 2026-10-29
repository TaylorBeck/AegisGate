"use client";

import { Card } from "@/components/ui/card";
import { useTelemetryContext } from "./telemetry-provider";

export function KPIRibbon() {
  const { state } = useTelemetryContext();
  const { kpis } = state;

  const l1Percent = (kpis.hitRatioL1 * 100).toFixed(1);
  const l2Percent = (kpis.hitRatioL2 * 100).toFixed(1);
  const combinedPercent = (kpis.combinedHitRatio * 100).toFixed(1);
  const p95Overhead = kpis.p95GatewayOverheadMs.toFixed(2);
  const avgTtft = kpis.avgUpstreamTtftMs.toFixed(1);
  const totalSaved = kpis.totalSavedUsd.toFixed(4);
  const totalActual = kpis.totalActualUsd.toFixed(4);

  const overheadUnderBudget = kpis.p95GatewayOverheadMs <= 12.0;

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            FinOps Saved
          </div>
          <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-emerald-400">
            ${totalSaved}
          </div>
        </div>
        <div className="mt-2 text-[10px] font-mono text-muted-foreground tabular-nums">
          Spent: ${totalActual}
        </div>
      </Card>

      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            L1 Exact Hits
          </div>
          <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-emerald-400">
            {l1Percent}%
          </div>
        </div>
        <div className="mt-2 text-[10px] font-mono text-muted-foreground">
          &lt; 2ms latency target
        </div>
      </Card>

      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            L2 Semantic Hits
          </div>
          <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-cyan-400">
            {l2Percent}%
          </div>
        </div>
        <div className="mt-2 text-[10px] font-mono text-muted-foreground">
          &ge; 0.94 cosine sim
        </div>
      </Card>

      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Combined Hit Rate
          </div>
          <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-foreground">
            {combinedPercent}%
          </div>
        </div>
        <div className="mt-2 space-y-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/40 flex">
            <div
              className="h-full bg-emerald-400 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, kpis.hitRatioL1 * 100))}%` }}
              title={`L1: ${l1Percent}%`}
            />
            <div
              className="h-full bg-cyan-400 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, kpis.hitRatioL2 * 100))}%` }}
              title={`L2: ${l2Percent}%`}
            />
          </div>
          <div className="text-[10px] font-mono text-muted-foreground flex justify-between">
            <span>L1 + L2 cache</span>
            <span>{combinedPercent}%</span>
          </div>
        </div>
      </Card>

      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            p95 Overhead
          </div>
          <div
            className={`mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums ${
              overheadUnderBudget ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {p95Overhead}ms
          </div>
        </div>
        <div className="mt-2 space-y-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/40">
            <div
              className={`h-full transition-all duration-300 ${
                overheadUnderBudget ? "bg-emerald-400" : "bg-rose-400"
              }`}
              style={{
                width: `${Math.min(100, Math.max(5, (kpis.p95GatewayOverheadMs / 12.0) * 50))}%`,
              }}
            />
          </div>
          <div className="text-[10px] font-mono text-muted-foreground flex justify-between">
            <span>Budget &lt; 12.0ms</span>
            <span className={overheadUnderBudget ? "text-emerald-400" : "text-rose-400 font-semibold"}>
              {overheadUnderBudget ? "IN SLA" : "BREACH"}
            </span>
          </div>
        </div>
      </Card>

      <Card className="border-border/50 bg-card/60 p-3.5 shadow-xs backdrop-blur-md hover:border-border/80 transition-all rounded-xl flex flex-col justify-between">
        <div>
          <div className="font-sans text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Avg Upstream TTFT
          </div>
          <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-foreground">
            {avgTtft}ms
          </div>
        </div>
        <div className="mt-2 text-[10px] font-mono text-muted-foreground tabular-nums">
          {kpis.totalRequests} total requests
        </div>
      </Card>
    </div>
  );
}


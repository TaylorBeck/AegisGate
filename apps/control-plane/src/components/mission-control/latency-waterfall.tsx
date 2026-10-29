"use client";

import { type LatencyBreakdown } from "@/lib/contracts";

interface LatencyWaterfallProps {
  breakdown: LatencyBreakdown;
  gatewayOverheadMs: number;
}

interface StageInfo {
  name: string;
  key: keyof LatencyBreakdown;
  description: string;
}

const STAGES: StageInfo[] = [
  {
    name: "Guardrail",
    key: "guardrail",
    description: "Compiled regex PII/secret scrubbing",
  },
  {
    name: "L1 Hash",
    key: "l1_lookup",
    description: "BLAKE3 key derivation & Lua admission lookup",
  },
  {
    name: "ONNX Embed",
    key: "onnx_embed",
    description: "In-process quantized bge-small ONNX vector inference",
  },
  {
    name: "Redis HNSW",
    key: "l2_hnsw",
    description: "FT.SEARCH KNN cosine vector distance query",
  },
  {
    name: "SimSIMD Route",
    key: "router",
    description: "Zero-shot complexity matrix classification & affinity",
  },
];

export function LatencyWaterfall({
  breakdown,
  gatewayOverheadMs,
}: LatencyWaterfallProps) {
  const maxMs = Math.max(gatewayOverheadMs, 1.0);

  return (
    <div className="space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-border/40 pb-2">
        <span className="font-semibold text-foreground">Hot-Path Stage Breakdown</span>
        <span className="text-muted-foreground">
          Total Overhead:{" "}
          <strong className="text-foreground">{gatewayOverheadMs.toFixed(2)}ms</strong>
        </span>
      </div>

      <div className="space-y-2.5">
        {STAGES.map((stage) => {
          const value = breakdown[stage.key];
          const isSkipped = value === null || value === undefined;
          const percent = isSkipped ? 0 : Math.min((value / maxMs) * 100, 100);

          return (
            <div key={stage.name} className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="font-semibold text-foreground">{stage.name}</span>
                  <span className="text-[10px] text-muted-foreground hidden sm:inline">
                    ({stage.description})
                  </span>
                </div>
                {isSkipped ? (
                  <span className="rounded bg-muted/40 px-1.5 py-0.5 text-[10px] text-muted-foreground italic">
                    skipped
                  </span>
                ) : (
                  <span className="font-bold tabular-nums text-foreground">
                    {value.toFixed(2)}ms
                  </span>
                )}
              </div>

              <div className="h-2 w-full overflow-hidden rounded bg-secondary/30">
                {!isSkipped && (
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-300"
                    style={{ width: `${Math.max(percent, 2)}%` }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


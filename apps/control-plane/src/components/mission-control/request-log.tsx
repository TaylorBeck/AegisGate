"use client";

import { ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { type RequestCompletedEvent } from "@/lib/contracts";
import { useTelemetryContext } from "./telemetry-provider";

export function RequestLog() {
  const { state, setSelectedRequest } = useTelemetryContext();

  const completedRequests: RequestCompletedEvent[] = state.events.filter(
    (e): e is RequestCompletedEvent => e.type === "request.completed",
  );

  return (
    <div className="rounded-xl border border-border/50 bg-card/50 shadow-xs backdrop-blur-md overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 px-4 py-3 bg-muted/20">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm text-foreground font-sans">
            Real-Time Inference Log
          </span>
          <span className="text-xs text-muted-foreground font-sans">
            ({completedRequests.length} cached in ring buffer)
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs font-sans">
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" /> L1 Exact (&lt;2ms)
          </span>
          <span className="flex items-center gap-1.5 text-cyan-400">
            <span className="h-2 w-2 rounded-full bg-cyan-400" /> L2 Semantic (&lt;15ms)
          </span>
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-zinc-500" /> Cache Miss
          </span>
        </div>
      </div>

      <div className="max-h-[520px] overflow-y-auto">
        <Table>
          <TableHeader className="bg-muted/30 sticky top-0 z-10 backdrop-blur-md">
            <TableRow className="border-border/40 hover:bg-transparent">
              <TableHead className="w-[100px] text-xs font-sans font-medium">Cache</TableHead>
              <TableHead className="w-[95px] text-xs font-sans font-medium">Time</TableHead>
              <TableHead className="text-xs font-sans font-medium min-w-[280px]">Scrubbed Prompt</TableHead>
              <TableHead className="w-[160px] text-xs font-sans font-medium">Model &amp; Tier</TableHead>
              <TableHead className="w-[95px] text-xs font-sans font-medium text-right">Overhead</TableHead>
              <TableHead className="w-[105px] text-xs font-sans font-medium text-right">Saved ($)</TableHead>
              <TableHead className="w-[40px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {completedRequests.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={7}
                  className="h-32 text-center text-muted-foreground text-xs font-sans"
                >
                  No requests received yet. Click{" "}
                  <strong className="text-foreground">&quot;Simulate 50 Requests&quot;</strong> or{" "}
                  <strong className="text-foreground">&quot;Test Paraphrase Cache Hit&quot;</strong> above to see live inference traffic.
                </TableCell>
              </TableRow>
            ) : (
              completedRequests.map((req) => {
                const finops = req.finops;
                const status = finops.cache_status;
                const timeStr = new Date(req.ts).toLocaleTimeString();

                return (
                  <TableRow
                    key={`${req.request_id}-${req.ts}`}
                    onClick={() => setSelectedRequest(req)}
                    className="group border-border/30 hover:bg-muted/40 cursor-pointer transition-colors"
                  >
                    <TableCell className="py-2.5">
                      {status === "HIT_L1" && (
                        <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30 text-[10px] font-mono font-semibold px-2 py-0.5">
                          HIT_L1
                        </Badge>
                      )}
                      {status === "HIT_L2" && (
                        <Badge className="bg-cyan-500/15 text-cyan-400 border-cyan-500/30 text-[10px] font-mono font-semibold px-2 py-0.5">
                          HIT_L2
                        </Badge>
                      )}
                      {status === "MISS" && (
                        <Badge
                          variant="outline"
                          className="border-border/60 bg-muted/40 text-muted-foreground text-[10px] font-mono font-normal px-2 py-0.5"
                        >
                          MISS
                        </Badge>
                      )}
                    </TableCell>

                    <TableCell className="text-muted-foreground text-xs font-mono tabular-nums py-2.5">
                      {timeStr}
                    </TableCell>

                    <TableCell className="font-sans text-xs text-foreground/90 max-w-lg truncate py-2.5">
                      {req.prompt_preview}
                    </TableCell>

                    <TableCell className="py-2.5">
                      <div className="flex flex-col gap-1 items-start">
                        <span className="font-mono text-xs font-medium text-foreground truncate">
                          {finops.routed_model}
                        </span>
                        <span className="rounded bg-muted/60 border border-border/40 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                          {finops.routed_tier}
                        </span>
                      </div>
                    </TableCell>

                    <TableCell className="text-right text-xs font-mono tabular-nums py-2.5">
                      <span
                        className={
                          finops.gateway_overhead_ms <= 12.0
                            ? "text-foreground font-medium"
                            : "text-rose-400 font-semibold"
                        }
                      >
                        {finops.gateway_overhead_ms.toFixed(1)}ms
                      </span>
                    </TableCell>

                    <TableCell className="text-right text-xs font-mono tabular-nums text-emerald-400 font-medium py-2.5">
                      +${finops.cost_saved_usd.toFixed(4)}
                    </TableCell>

                    <TableCell className="py-2.5 text-right w-[40px] pr-3">
                      <ChevronRight className="h-4 w-4 text-muted-foreground/30 group-hover:text-foreground group-hover:translate-x-0.5 transition-all inline" />
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

"use client";

import { Badge } from "@/components/ui/badge";
import { type BreakerName } from "@/lib/contracts";
import { useTelemetryContext } from "./telemetry-provider";

export function BreakerStatus() {
  const { state } = useTelemetryContext();

  const knownProviders = ["mock-primary", "mock-secondary"];
  // Merge any dynamic providers seen in state.breakers
  const providers = Array.from(
    new Set([...knownProviders, ...Object.keys(state.breakers)]),
  );

  return (
    <div className="flex flex-wrap items-center gap-2.5">
      <span className="text-xs font-medium text-muted-foreground font-sans">
        Circuit Breakers:
      </span>
      {providers.map((provider) => {
        const breakerState: BreakerName = state.breakers[provider] ?? "CLOSED";
        const isClosed = breakerState === "CLOSED";
        const isOpen = breakerState === "OPEN";
        const isHalfOpen = breakerState === "HALF_OPEN";

        return (
          <div
            key={provider}
            className="flex items-center gap-2 rounded-lg border border-border/40 bg-card/60 px-2.5 py-1 text-xs"
          >
            <span className="font-mono text-[11px] text-foreground/90">{provider}</span>
            {isClosed && (
              <Badge
                variant="outline"
                className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-mono text-[10px] px-1.5 py-0"
              >
                CLOSED
              </Badge>
            )}
            {isOpen && (
              <Badge
                variant="destructive"
                className="animate-pulse bg-red-600 font-mono text-[10px] text-white px-1.5 py-0"
              >
                OPEN
              </Badge>
            )}
            {isHalfOpen && (
              <Badge
                variant="outline"
                className="border-amber-500/30 bg-amber-500/10 text-amber-400 font-mono text-[10px] px-1.5 py-0"
              >
                HALF_OPEN
              </Badge>
            )}
          </div>
        );
      })}
    </div>
  );
}


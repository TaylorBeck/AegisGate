"use client";

import { Activity, ShieldAlert, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTelemetryContext } from "./telemetry-provider";

export function SimulationActions() {
  const { runSimulation, loadingScenarios, state } = useTelemetryContext();

  const isBurstRunning =
    loadingScenarios.burst ||
    Object.values(state.activeRuns).some(
      (r) => r.scenario === "burst" && !r.completed,
    );
  const isParaphraseRunning =
    loadingScenarios.paraphrase ||
    Object.values(state.activeRuns).some(
      (r) => r.scenario === "paraphrase" && !r.completed,
    );
  const isOutageRunning =
    loadingScenarios.outage ||
    Object.values(state.activeRuns).some(
      (r) => r.scenario === "outage" && !r.completed,
    );

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="sm"
        disabled={isBurstRunning}
        onClick={() => runSimulation("burst")}
        className="h-8 px-3.5 text-xs font-sans font-medium bg-indigo-600 hover:bg-indigo-500 text-white shadow-xs transition-colors cursor-pointer"
      >
        <Zap className={`mr-1.5 h-3.5 w-3.5 ${isBurstRunning ? "animate-spin" : ""}`} />
        {isBurstRunning ? "Simulating..." : "Simulate 50 Requests"}
      </Button>

      <Button
        variant="outline"
        size="sm"
        disabled={isParaphraseRunning}
        onClick={() => runSimulation("paraphrase")}
        className="h-8 px-3 text-xs font-sans font-medium border-border/60 text-foreground/90 hover:bg-muted/50 hover:text-foreground transition-colors cursor-pointer"
      >
        <Activity className={`mr-1.5 h-3.5 w-3.5 ${isParaphraseRunning ? "animate-pulse" : ""}`} />
        {isParaphraseRunning ? "Testing Hit..." : "Test Paraphrase Cache Hit"}
      </Button>

      <Button
        variant="outline"
        size="sm"
        disabled={isOutageRunning}
        onClick={() => runSimulation("outage")}
        className="h-8 px-3 text-xs font-sans font-medium border-rose-500/35 text-rose-400 hover:bg-rose-950/30 hover:border-rose-500/60 hover:text-rose-300 transition-colors cursor-pointer"
      >
        <ShieldAlert className={`mr-1.5 h-3.5 w-3.5 ${isOutageRunning ? "animate-bounce" : ""}`} />
        {isOutageRunning ? "Failing Over..." : "Trigger Provider Outage"}
      </Button>
    </div>
  );
}

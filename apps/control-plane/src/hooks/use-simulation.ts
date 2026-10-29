"use client";

import { useCallback, useState } from "react";

export type ScenarioType = "burst" | "paraphrase" | "outage";

export interface SimulationRun {
  runId: string;
  scenario: ScenarioType;
  startTime: number;
  status: "running" | "completed" | "error";
  error?: string;
}

export function useSimulation(gatewayUrl?: string) {
  const [activeRuns, setActiveRuns] = useState<Record<string, SimulationRun>>({});
  const [loadingScenarios, setLoadingScenarios] = useState<Record<ScenarioType, boolean>>({
    burst: false,
    paraphrase: false,
    outage: false,
  });

  const runSimulation = useCallback(
    async (scenario: ScenarioType): Promise<string | null> => {
      const baseUrl =
        gatewayUrl ||
        process.env.NEXT_PUBLIC_GATEWAY_URL ||
        "http://localhost:8000";
      const url = `${baseUrl.replace(/\/$/, "")}/v1/aegis/simulate/${scenario}`;

      setLoadingScenarios((prev) => ({ ...prev, [scenario]: true }));

      try {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!res.ok) {
          const errData = (await res.json().catch(() => ({}))) as {
            error?: { message?: string };
          };
          const errMsg =
            errData?.error?.message ||
            `Failed to trigger scenario: ${res.statusText}`;
          throw new Error(errMsg);
        }

        const data = (await res.json()) as { run_id: string };
        const runId = data.run_id;

        setActiveRuns((prev) => ({
          ...prev,
          [runId]: {
            runId,
            scenario,
            startTime: Date.now(),
            status: "running",
          },
        }));

        return runId;
      } catch (err) {
        console.error(`Simulation trigger failed for ${scenario}:`, err);
        return null;
      } finally {
        setLoadingScenarios((prev) => ({ ...prev, [scenario]: false }));
      }
    },
    [gatewayUrl],
  );

  return {
    runSimulation,
    activeRuns,
    loadingScenarios,
  };
}


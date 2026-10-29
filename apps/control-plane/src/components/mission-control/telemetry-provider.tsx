"use client";

import React, { createContext, useContext, useState } from "react";
import { type RequestCompletedEvent } from "@/lib/contracts";
import { type TelemetryAction, type TelemetryState } from "@/lib/telemetry-store";
import { type ScenarioType, useSimulation } from "@/hooks/use-simulation";
import { useTelemetry } from "@/hooks/use-telemetry";

interface TelemetryContextValue {
  state: TelemetryState;
  dispatch: React.Dispatch<TelemetryAction>;
  isConnected: boolean;
  connectionError: string | null;
  selectedRequest: RequestCompletedEvent | null;
  setSelectedRequest: (req: RequestCompletedEvent | null) => void;
  runSimulation: (scenario: ScenarioType) => Promise<string | null>;
  loadingScenarios: Record<ScenarioType, boolean>;
}

const TelemetryContext = createContext<TelemetryContextValue | null>(null);

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const { state, dispatch, isConnected, connectionError } = useTelemetry();
  const { runSimulation, loadingScenarios } = useSimulation();
  const [selectedRequest, setSelectedRequest] =
    useState<RequestCompletedEvent | null>(null);

  return (
    <TelemetryContext.Provider
      value={{
        state,
        dispatch,
        isConnected,
        connectionError,
        selectedRequest,
        setSelectedRequest,
        runSimulation,
        loadingScenarios,
      }}
    >
      {children}
    </TelemetryContext.Provider>
  );
}

export function useTelemetryContext(): TelemetryContextValue {
  const ctx = useContext(TelemetryContext);
  if (!ctx) {
    throw new Error(
      "useTelemetryContext must be used within a TelemetryProvider",
    );
  }
  return ctx;
}


"use client";

import { useTelemetryContext } from "./telemetry-provider";

export function ConnectionBadge() {
  const { isConnected, connectionError } = useTelemetryContext();

  if (isConnected) {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-xs font-sans font-medium text-emerald-400">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
        </span>
        Live Connected
      </div>
    );
  }

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-amber-500/25 bg-amber-500/10 px-2.5 py-1 text-xs font-sans font-medium text-amber-400"
      title={connectionError ?? ""}
    >
      <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
      Connecting...
    </div>
  );
}


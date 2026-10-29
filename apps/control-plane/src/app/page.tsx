import Image from "next/image";
import { BreakerStatus } from "@/components/mission-control/breaker-status";
import { ConnectionBadge } from "@/components/mission-control/connection-badge";
import { KPIRibbon } from "@/components/mission-control/kpi-ribbon";
import { RequestLog } from "@/components/mission-control/request-log";
import { SimulationActions } from "@/components/mission-control/simulation-actions";
import { TelemetryProvider } from "@/components/mission-control/telemetry-provider";
import { TraceDrawer } from "@/components/mission-control/trace-drawer";

export default function MissionControlPage() {
  return (
    <TelemetryProvider>
      <div className="min-h-screen flex flex-col bg-background text-foreground">
        {/* Header */}
        <header className="border-b border-border/40 bg-card/60 backdrop-blur-md sticky top-0 z-20">
          <div className="container mx-auto flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <Image
                src="/icon.png"
                alt="AegisGate Logo"
                width={36}
                height={36}
                className="h-9 w-9 rounded-lg object-contain shadow-md shadow-indigo-500/20"
                priority
              />
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-sm font-semibold tracking-tight text-foreground font-sans">
                    AegisGate Mission Control
                  </h1>
                  <span className="rounded-full border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px] font-mono text-primary font-medium">
                    v0.1.0
                  </span>
                </div>
                <p className="text-xs text-muted-foreground font-sans">
                  Ultra-low-latency Semantic LLM Gateway &amp; Cloud FinOps Proxy
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <ConnectionBadge />
              <SimulationActions />
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="container mx-auto flex-1 space-y-6 px-4 py-6">
          {/* Status Sub-bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border/30 pb-4">
            <BreakerStatus />
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground font-sans">
              <span className="font-medium text-muted-foreground/80">Hot-Path SLA:</span>
              <div className="flex items-center gap-1.5 font-mono text-[11px]">
                <span className="rounded bg-muted/60 px-1.5 py-0.5 border border-border/40 text-foreground font-semibold">
                  p95 &lt;12ms
                </span>
                <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 border border-emerald-500/20 text-emerald-400 font-semibold">
                  L1 &lt;2ms
                </span>
                <span className="rounded bg-cyan-500/10 px-1.5 py-0.5 border border-cyan-500/20 text-cyan-400 font-semibold">
                  L2 &lt;15ms (cos &ge; 0.94)
                </span>
              </div>
            </div>
          </div>

          {/* KPI Ribbon */}
          <section aria-label="FinOps Performance Overview">
            <KPIRibbon />
          </section>

          {/* Real-Time Request Stream */}
          <section aria-label="Inference Request Stream">
            <RequestLog />
          </section>
        </main>

        {/* Footer */}
        <footer className="border-t border-border/40 bg-card/20 py-3 text-center text-xs text-muted-foreground font-sans">
          AegisGate Zero-Auth Control Plane · Quantized ONNX FastEmbed · Valkey HNSW · HTTP/2 Multiplexing
        </footer>

        {/* Trace Drawer */}
        <TraceDrawer />
      </div>
    </TelemetryProvider>
  );
}

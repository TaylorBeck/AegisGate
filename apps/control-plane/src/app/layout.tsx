import type { Metadata } from "next";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

export const metadata: Metadata = {
  title: "AegisGate — Semantic LLM Gateway & Cloud FinOps Proxy",
  description: "Mission Control for ultra-low latency semantic LLM routing and vector caching",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-background text-foreground font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}

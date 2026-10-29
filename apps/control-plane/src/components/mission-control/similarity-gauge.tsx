"use client";

interface SimilarityGaugeProps {
  score: number | null | undefined;
  threshold?: number;
}

export function SimilarityGauge({
  score,
  threshold = 0.94,
}: SimilarityGaugeProps) {
  if (score === null || score === undefined) {
    return (
      <div className="rounded-lg border border-border/40 bg-card/30 p-3 text-center text-xs text-muted-foreground">
        Semantic L2 search skipped for this request
      </div>
    );
  }

  const isHit = score >= threshold;
  const percent = Math.min(Math.max(score * 100, 0), 100);
  const thresholdPercent = threshold * 100;

  return (
    <div className="space-y-2 rounded-lg border border-border/40 bg-card/30 p-3">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-muted-foreground">Cosine Similarity:</span>
        <span
          className={`font-bold tabular-nums ${
            isHit ? "text-cyan-400" : "text-amber-400"
          }`}
        >
          {score.toFixed(4)} {isHit ? "(HIT ≥ 0.94)" : "(MISS < 0.94)"}
        </span>
      </div>

      <div className="relative h-3 w-full overflow-hidden rounded-full bg-secondary/50">
        <div
          className={`h-full transition-all duration-300 ${
            isHit ? "bg-gradient-to-r from-cyan-600 to-cyan-400" : "bg-gradient-to-r from-amber-600 to-amber-400"
          }`}
          style={{ width: `${percent}%` }}
        />
        {/* Threshold marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-sm shadow-white"
          style={{ left: `${thresholdPercent}%` }}
          title={`Threshold: ${threshold}`}
        />
      </div>

      <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
        <span>0.00</span>
        <span className="text-foreground/80 font-semibold">0.94 (Cache Threshold)</span>
        <span>1.00</span>
      </div>
    </div>
  );
}


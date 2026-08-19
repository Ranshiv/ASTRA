import { Activity, AlertTriangle, Cpu, Database, ListChecks, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { CacheStatus, DeviceReport } from "@/lib/engine";

type Tone = "ok" | "warn" | "bad" | "neutral";

const DOT_TONE: Record<Tone, string> = {
  ok: "bg-[var(--color-ok)]",
  warn: "bg-[var(--color-warn)]",
  bad: "bg-[var(--color-bad)]",
  neutral: "bg-[var(--color-muted)]",
};

function Tile({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral",
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail?: string;
  tone?: Tone;
}) {
  return (
    <div className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-3">
      <div className="flex items-center gap-2 text-[var(--color-muted)]">
        <Icon size={14} strokeWidth={2} />
        <span className="text-xs">{label}</span>
        <span className={`ml-auto h-1.5 w-1.5 rounded-full ${DOT_TONE[tone]}`} />
      </div>
      <p className="mt-1.5 truncate text-sm font-medium text-[var(--color-text)]" title={value}>
        {value}
      </p>
      {detail && <p className="mt-0.5 truncate text-[11px] text-[var(--color-muted)]">{detail}</p>}
    </div>
  );
}

export function StatusStrip({
  device,
  cache,
  error,
  curvesCount,
  candidatesCount,
}: {
  device: DeviceReport | null;
  cache: CacheStatus | null;
  error: string | null;
  curvesCount?: number;
  candidatesCount?: number;
}) {
  if (error) {
    return (
      <div className="flex items-start gap-3 rounded-lg border border-[var(--color-bad)]/40 bg-[var(--color-bad)]/5 p-4">
        <AlertTriangle size={16} className="mt-0.5 shrink-0 text-[var(--color-bad)]" />
        <div>
          <p className="text-sm text-[var(--color-bad)]">Scientific engine unavailable</p>
          <p className="mt-1 font-mono text-xs text-[var(--color-muted)]">{error}</p>
        </div>
      </div>
    );
  }

  const gpuTone: Tone = device?.device === "cuda" ? "ok" : "warn";
  const vramLow = (device?.gpu?.free_vram_mb ?? Infinity) < 1500;
  const cachePressure = (cache?.usage_fraction ?? 0) > 0.8;

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile
          icon={Cpu}
          label="Engine"
          value={device ? "Ready" : "Starting…"}
          tone={device ? "ok" : "neutral"}
        />
        <Tile
          icon={Zap}
          label="Execution mode"
          value={device ? device.device.toUpperCase() : "—"}
          detail={device?.gpu?.name}
          tone={device ? gpuTone : "neutral"}
        />
        {device?.gpu && (
          <Tile
            icon={Database}
            label="VRAM free"
            value={`${device.gpu.free_vram_mb} MB`}
            detail={`of ${device.gpu.total_vram_mb} MB`}
            tone={vramLow ? "warn" : "ok"}
          />
        )}
        {cache && (
          <Tile
            icon={Database}
            label="Cache"
            value={`${cache.total_gb.toFixed(2)} GB`}
            detail={`of ${cache.cap_gb.toFixed(0)} GB cap`}
            tone={cachePressure ? "warn" : "ok"}
          />
        )}
        {typeof curvesCount === "number" && (
          <Tile icon={Activity} label="Light curves" value={curvesCount.toLocaleString()} tone="neutral" />
        )}
        {typeof candidatesCount === "number" && (
          <Tile icon={ListChecks} label="Candidates" value={candidatesCount.toLocaleString()} tone="neutral" />
        )}
      </div>

      {device && (
        <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted)]">{device.reason}</p>
      )}
    </div>
  );
}

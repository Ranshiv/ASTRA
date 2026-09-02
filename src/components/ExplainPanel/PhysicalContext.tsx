import { useState } from "react";
import { Thermometer } from "lucide-react";

import { engine, type Candidate, type PhysicalCharacterization } from "@/lib/engine";
import { Badge, Button, Empty, KeyValue, Note } from "@/components/ui";

export function PhysicalContext({ candidate }: { candidate: Candidate }) {
  const [physical, setPhysical] = useState<PhysicalCharacterization | null>(
    candidate.physical_characterization as PhysicalCharacterization | undefined ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function characterize() {
    setBusy(true);
    setError(null);
    try {
      setPhysical(await engine.physicalCharacterize(candidate.features));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }
  if (!physical) {
    return (
      <div className="flex flex-col gap-2">
        <Empty>No broadband physical characterization has been run for this candidate.</Empty>
        <Button icon={Thermometer} disabled={busy} onClick={() => void characterize()}>
          {busy ? "Characterizing…" : "Characterize available photometry"}
        </Button>
        {error && <Note tone="bad">{error}</Note>}
      </div>
    );
  }
  const temperature = physical.temperature_k;
  const bands = Array.isArray(physical.bands_used) ? physical.bands_used.join(", ") : "—";
  const colors = physical.colors && typeof physical.colors === "object"
    ? physical.colors as Record<string, unknown> : null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={physical.quality === "usable" ? "ok" : "warn"}>{String(physical.quality ?? "unknown")}</Badge>
        <Badge tone="muted">{temperature == null ? "temperature unavailable" : `${Number(temperature).toFixed(0)} K`}</Badge>
        <Badge tone="muted">bands: {bands}</Badge>
        <Button icon={Thermometer} disabled={busy} onClick={() => void characterize()}>
          {busy ? "Updating…" : "Recalculate"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {colors && <KeyValue rows={Object.entries(colors).map(([key, value]) => [key, String(value)] as [string, string])} />}
      {Array.isArray(physical.warnings) && physical.warnings.length > 0 && (
        <ul className="list-inside list-disc text-xs text-[var(--color-muted)]">
          {physical.warnings.map((warning, index) => <li key={index}>{String(warning)}</li>)}
        </ul>
      )}
      <Note tone="warn">This is a bounded broadband SED diagnostic, not a spectral classification or extinction measurement.</Note>
    </div>
  );
}

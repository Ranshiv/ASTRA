import { useState } from "react";
import { Microscope } from "lucide-react";

import { engine, type AttributionResult, type Candidate } from "@/lib/engine";
import { Badge, Button, Note, Table, num } from "@/components/ui";

/** Occlusion-based feature attribution for the raw anomaly ensemble score --
 * a different, deeper question than ScoreDrivers, which explains the
 * composite score's already-transparent weights. Reruns the ensemble once
 * per feature, so this is an explicit, busy-gated research action. */
export function FeatureAttribution({ candidate, projectId }: { candidate: Candidate; projectId?: string }) {
  const [result, setResult] = useState<AttributionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stable, setStable] = useState(false);

  async function explain() {
    setBusy(true);
    setError(null);
    try {
      setResult(await engine.candidateExplain(candidate.candidate_id, "default", projectId, 10, stable));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <Button icon={Microscope} disabled={busy} onClick={() => void explain()}>
          {busy
            ? stable
              ? "Explaining… (reruns top features against multiple typical values)"
              : "Explaining… (reruns the ensemble per feature)"
            : "Explain (feature attribution)"}
        </Button>
        <label className="flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
          <input
            type="checkbox"
            checked={stable}
            disabled={busy}
            onChange={(event) => setStable(event.target.checked)}
          />
          Include confidence (slower — reruns top features against multiple typical values)
        </label>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {result && !result.explainable && (
        <Note tone="warn">{result.reason ?? "This candidate cannot be explained."}</Note>
      )}
      {result?.explainable && result.components && (
        <>
          <Note tone="muted">
            Baseline consensus score {num(result.baseline_score, 4)}. Each row shows how much the
            score would drop if that feature were replaced by its typical (population median) value.
          </Note>
          {result.narrative && <Note tone="ok">{result.narrative}</Note>}
          <Table head={["Feature", "Value", "Typical", "Impact"]}>
            {result.components.map((component) => (
              <tr key={component.feature} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5">
                  {component.label ?? component.feature.replace(/_/g, " ")}
                  {component.unit ? <span className="text-[var(--color-muted)]"> ({component.unit})</span> : null}
                </td>
                <td className="px-2 py-1.5">{num(component.value, 3)}</td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">{num(component.typical, 3)}</td>
                <td className="px-2 py-1.5">
                  <div className="flex flex-wrap items-center gap-1">
                    <Badge tone={component.impact > 0 ? "bad" : component.impact < 0 ? "ok" : "muted"}>
                      {component.impact > 0 ? "raises" : component.impact < 0 ? "suppresses" : "neutral"}{" "}
                      {num(component.impact, 4)}
                    </Badge>
                    {component.stable !== undefined && (
                      <Badge tone={component.stable ? "ok" : "warn"}>
                        {component.stable ? "stable" : "reference-sensitive"}
                      </Badge>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        </>
      )}
    </div>
  );
}

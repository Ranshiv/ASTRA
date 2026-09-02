import type { Candidate } from "@/lib/engine";
import { Badge, Empty, Table, num } from "@/components/ui";

/** The artifact case against a candidate, and the evidence clearing it. */
export function ArtifactCase({ candidate }: { candidate: Candidate }) {
  const assessment = candidate.explanation.could_be_artifact ?? candidate.artifact ?? {};
  const indicators = assessment.indicators ?? [];
  const clearing = assessment.clearing_evidence ?? [];
  const likelihood = assessment.likelihood ?? 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={likelihood >= 0.6 ? "bad" : likelihood >= 0.3 ? "warn" : "ok"}>
          {assessment.verdict ?? "not assessed"}
        </Badge>
        <Badge tone="muted">likelihood {num(assessment.likelihood, 3)}</Badge>
      </div>

      {indicators.length === 0 ? (
        <Empty>No artifact indicators fired for this candidate.</Empty>
      ) : (
        <Table head={["Indicator", "Weight", "Why it fired"]}>
          {indicators.map((indicator) => (
            <tr key={indicator.name} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{indicator.name.replace(/_/g, " ")}</td>
              <td className="px-2 py-1.5">{num(indicator.weight, 2)}</td>
              <td className="px-2 py-1.5 text-[var(--color-muted)]">{indicator.detail}</td>
            </tr>
          ))}
        </Table>
      )}

      {clearing.length > 0 && (
        <>
          <p className="text-[11px] text-[var(--color-muted)]">Evidence against an artifact</p>
          <ul className="list-inside list-disc text-xs text-[var(--color-ok)]">
            {clearing.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </>
      )}
    </div>
  );
}

import type { Candidate } from "@/lib/engine";
import { Empty, KeyValue, Note } from "@/components/ui";

export function SignificanceContext({ candidate }: { candidate: Candidate }) {
  const report = candidate.significance;
  const completeness = candidate.evidence_completeness;
  if (!report && !completeness) {
    return <Empty>No calibrated significance or evidence-completeness report is attached yet.</Empty>;
  }
  return (
    <div className="flex flex-col gap-2">
      {report && (
        <KeyValue rows={[
          ["Calibration", report.reference_kind ?? report.method],
          ["Tail probability", report.tail_probability !== undefined
            ? `${(report.tail_probability * 100).toFixed(2)}%`
            : report.tail_probability_summary
            ? `${(report.tail_probability_summary.min * 100).toFixed(2)}%–${(report.tail_probability_summary.max * 100).toFixed(2)}%`
            : "—"],
          ["Estimated FDR", report.estimated_fdr === undefined ? "—" : `${(report.estimated_fdr * 100).toFixed(1)}%`],
          ["Reference population", report.n_reference],
        ]} />
      )}
      {completeness && (
        <Note tone="muted">
          Evidence completeness: {String(completeness.available ?? completeness.resolved ?? "unknown")}
          {completeness.total ? ` of ${String(completeness.total)} sources` : ""}.
        </Note>
      )}
      <Note tone="warn">
        Calibration is an interpretation layer; it does not change the historical ASTRA ranking score.
      </Note>
    </div>
  );
}

import type { Candidate } from "@/lib/engine";
import { Badge, Empty, Note, Table, num } from "@/components/ui";

/** GW coincidence, shown but never folded into the score.
 *
 * `gw.py` deliberately does not touch scoring.WEIGHTS: this is unvalidated
 * evidence with no track record in this project yet, the same restraint
 * applied to the calibrated artifact weights. A candidate must not read as
 * "ranked higher because of this" -- it reads as "here is what was checked."
 */
export function GwCoincidence({ candidate }: { candidate: Candidate }) {
  const evidence = candidate.gw;
  if (!evidence) {
    return <Empty>No GW coincidence check has been run for this candidate yet.</Empty>;
  }
  if (evidence.state === "unavailable") {
    return <Note tone="warn">{evidence.reason ?? "GW coincidence could not be checked."}</Note>;
  }
  if (evidence.checked_events === 0) {
    return <Empty>No GW coincidence check has been run for this candidate yet.</Empty>;
  }

  const coincident = evidence.coincident ?? [];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={evidence.state === "match" ? "bad" : "ok"}>
          {evidence.state === "match" ? "coincidence found" : "no coincidence"}
        </Badge>
        <Badge tone="muted">{evidence.checked_events} events checked</Badge>
        {evidence.temporally_coincident !== undefined && (
          <Badge tone="muted">{evidence.temporally_coincident} temporally coincident</Badge>
        )}
      </div>

      {coincident.length === 0 ? (
        <Empty>No event overlapped this candidate's observation window.</Empty>
      ) : (
        <Table head={["Event", "Credible level", "In 90% region", "Position source"]}>
          {coincident.map((item) => (
            <tr key={item.event} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{item.event}</td>
              <td className="px-2 py-1.5">{num(item.credible_level, 3)}</td>
              <td className="px-2 py-1.5">
                <Badge tone={item.in_90pct_region ? "warn" : "muted"}>
                  {item.in_90pct_region ? "yes" : "no"}
                </Badge>
              </td>
              <td className="px-2 py-1.5 text-[var(--color-muted)]">
                {item.position_source === "em_counterpart_fixed"
                  ? "fixed to known counterpart"
                  : "GW posterior"}
              </td>
            </tr>
          ))}
        </Table>
      )}

      <Note>
        Not part of this candidate's score -- this evidence has no validated track
        record in this project yet.
      </Note>
    </div>
  );
}

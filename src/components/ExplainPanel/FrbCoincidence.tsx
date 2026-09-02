import type { Candidate } from "@/lib/engine";
import { Badge, Empty, Note, Table, num } from "@/components/ui";

/** FRB coincidence, shown but never folded into the score -- same restraint
 *  as GwCoincidence, for the identical reason (see frb.py's module docstring).
 *  Most bursts report an error-ellipse offset (sigma_offset), not a
 *  confidence-level fraction; only baseband-localized bursts (position_source
 *  "healpix") carry a true confidence_level. */
export function FrbCoincidence({ candidate }: { candidate: Candidate }) {
  const evidence = candidate.frb;
  if (!evidence) {
    return <Empty>No FRB coincidence check has been run for this candidate yet.</Empty>;
  }
  if (evidence.state === "unavailable") {
    return <Note tone="warn">{evidence.reason ?? "FRB coincidence could not be checked."}</Note>;
  }
  if (evidence.checked_bursts === 0) {
    return <Empty>No FRB coincidence check has been run for this candidate yet.</Empty>;
  }

  const coincident = evidence.coincident ?? [];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={evidence.state === "match" ? "bad" : "ok"}>
          {evidence.state === "match" ? "coincidence found" : "no coincidence"}
        </Badge>
        <Badge tone="muted">{evidence.checked_bursts} bursts checked</Badge>
        {evidence.temporally_coincident !== undefined && (
          <Badge tone="muted">{evidence.temporally_coincident} temporally coincident</Badge>
        )}
      </div>

      {coincident.length === 0 ? (
        <Empty>No burst overlapped this candidate's observation window.</Empty>
      ) : (
        <Table head={["Burst", "Offset / confidence", "Position source"]}>
          {coincident.map((item) => (
            <tr key={item.burst} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{item.burst}</td>
              <td className="px-2 py-1.5">
                {item.position_source === "healpix"
                  ? `CL ${num(item.confidence_level, 3)}`
                  : `${num(item.sigma_offset, 2)}σ`}
              </td>
              <td className="px-2 py-1.5 text-[var(--color-muted)]">
                {item.position_source === "healpix" ? "baseband localization" : "error ellipse"}
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

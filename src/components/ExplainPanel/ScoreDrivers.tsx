import type { Candidate } from "@/lib/engine";
import { Badge, Empty, Note, Table, num } from "@/components/ui";

/** Score components ordered by what they actually contributed, not by name. */
export function ScoreDrivers({ candidate }: { candidate: Candidate }) {
  const components = candidate.score.components ?? {};
  const weighted = candidate.score.weighted ?? {};
  const names = Object.keys(components).sort(
    (a, b) => (weighted[b] ?? 0) - (weighted[a] ?? 0),
  );

  if (names.length === 0) {
    return <Empty>This candidate carries no score components to explain.</Empty>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone="accent">total {num(candidate.score.total, 4)}</Badge>
        <Badge tone={(candidate.score.weight_used ?? 0) >= 0.999 ? "muted" : "warn"}>
          weight used {num(candidate.score.weight_used, 2)}
        </Badge>
        {candidate.score.ranking_method && (
          <Badge tone="muted">{candidate.score.ranking_method}</Badge>
        )}
      </div>

      <Table head={["Component", "Score", "Contribution"]}>
        {names.map((name) => (
          <tr key={name} className="border-b border-[var(--color-edge)]/50">
            <td className="px-2 py-1.5">{name.replace(/_/g, " ")}</td>
            <td className="px-2 py-1.5">{num(components[name], 4)}</td>
            <td className="px-2 py-1.5 text-[var(--color-accent)]">
              {num(weighted[name], 4)}
            </td>
          </tr>
        ))}
      </Table>

      {(candidate.score.weight_used ?? 1) < 0.999 && (
        <Note tone="warn">
          Some components could not be computed, so the total is renormalised over the
          weight that was available. It is not comparable with a fully weighted score.
        </Note>
      )}

      {(candidate.score.reasons?.length ?? 0) > 0 && (
        <ul className="list-inside list-disc text-xs text-[var(--color-muted)]">
          {candidate.score.reasons?.map((reason, index) => <li key={index}>{reason}</li>)}
        </ul>
      )}
    </div>
  );
}

import { ChevronRight, Layers } from "lucide-react";
import { useState } from "react";

import { Button, Note } from "@/components/ui";
import { engine, type Candidate, type DiscardRecord } from "@/lib/engine";

export function CandidateDiscardPilePanel({ candidate }: { candidate: Candidate }) {
  const [records, setRecords] = useState<DiscardRecord[] | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const isZtf = candidate.survey.toUpperCase() === "ZTF";

  async function scan() {
    setLoading(true);
    setStatus("Re-requesting the real IRSA endpoint with the server-side quality filter disabled…");
    try {
      const result = await engine.discardScan({
        objectId: candidate.object_id,
        raDeg: candidate.ra_deg,
        decDeg: candidate.dec_deg,
      });
      setRecords(result.records);
      setStatus(
        result.records.length
          ? `${result.records.length} coherent discarded-epoch run${result.records.length === 1 ? "" : "s"} found.`
          : "No coherent discarded-epoch run found — either nothing was flagged, or flagged runs were too short or too noisy.",
      );
    } catch (err) {
      setStatus(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <details className="group rounded border border-[var(--color-edge)] p-2">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs [&::-webkit-details-marker]:hidden">
        <ChevronRight
          size={12}
          strokeWidth={2}
          className="text-[var(--color-muted)] transition-transform duration-200 group-open:rotate-90"
        />
        <Layers size={13} strokeWidth={2} />
        Discard pile
      </summary>
      <p className="mt-1 text-xs text-[var(--color-muted)]">
        Epochs ZTF's own real per-epoch quality flags discard before this candidate is ever assembled, re-fetched
        directly from IRSA with the server-side filter disabled. A run is flagged coherent when it moves smoothly
        toward one excursion rather than jittering around the baseline — the discriminator between a real,
        under-flagged signal and a bad pixel or cosmic ray. This is a new, unproven signal: it is not part of this
        candidate's score.
      </p>
      {!isZtf && (
        <div className="mt-1.5">
          <Note>Discard-pile recovery only applies to ZTF — this candidate is {candidate.survey}.</Note>
        </div>
      )}
      {isZtf && (
        <div className="my-2">
          <Button onClick={() => void scan()} loading={loading}>
            Scan for discarded epochs
          </Button>
        </div>
      )}
      {status && <Note>{status}</Note>}
      {records && records.length > 0 && (
        <ul className="mt-2 flex flex-col gap-1.5">
          {records.map((record, index) => (
            <li
              key={`${record.time_start}-${index}`}
              className="rounded border border-[var(--color-edge)] p-1.5 text-xs"
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                <span className="font-medium">{record.flag_category}</span>
                <span className="text-[var(--color-muted)]">{record.epoch_count} epochs</span>
                <span className="text-[var(--color-muted)]">
                  {record.magnitude_offset > 0 ? "fainter" : "brighter"} by {Math.abs(record.magnitude_offset).toFixed(3)} mag
                </span>
                <span className={record.coherent ? "text-[var(--color-accent)]" : "text-[var(--color-muted)]"}>
                  {record.coherent ? "coherent excursion" : "not coherent"}
                </span>
              </div>
              <div className="mt-0.5 text-[var(--color-muted)]">
                {record.time_start.toFixed(3)} – {record.time_end.toFixed(3)} ({record.band})
              </div>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

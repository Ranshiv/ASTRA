import { useState } from "react";

import { Button } from "@/components/ui";
import { engine, type Candidate } from "@/lib/engine";

const LABELS = ["interesting", "artifact", "known_object", "uncertain", "needs_follow_up"] as const;

export function CandidateNotesPanel({
  candidate,
  projectId,
  onUpdated,
}: {
  candidate: Candidate;
  projectId?: string;
  onUpdated: (updated: Candidate) => void;
}) {
  const [note, setNote] = useState(candidate.review?.note ?? "");

  async function saveNote() {
    try {
      const labelValue = candidate.label ?? "uncertain";
      await engine.label(candidate.candidate_id, labelValue, note, projectId);
      onUpdated({ ...candidate, review: { label: labelValue, note, recorded_utc: new Date().toISOString() } });
    } catch {
      /* status is surfaced by the caller's refresh cycle */
    }
  }

  async function label(value: string) {
    await engine.label(candidate.candidate_id, value, note, projectId);
    onUpdated({
      ...candidate,
      label: value,
      review: { label: value, note, recorded_utc: new Date().toISOString() },
    });
  }

  return (
    <>
      <div className="rounded border border-[var(--color-edge)] p-2">
        <label className="block text-xs text-[var(--color-muted)]">
          Research note
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            rows={2}
            maxLength={4000}
            className="mt-1 min-h-9 w-full resize-y rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2.5 py-1.5 text-sm text-[var(--color-text)] outline-none transition-colors focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40"
            placeholder="Record why this candidate matters or what to check next."
          />
        </label>
        <div className="mt-2">
          <Button onClick={() => void saveNote()}>Save note</Button>
        </div>
      </div>

      {candidate.explanation.coverage?.status === "insufficient_data_lt_10_points" && (
        <p className="text-xs text-[var(--color-warn)]">Retained for review: fewer than 10 finite points.</p>
      )}

      <div className="flex flex-wrap gap-1.5">
        {LABELS.map((item) => (
          <button
            type="button"
            key={item}
            onClick={() => void label(item)}
            className={`rounded-full border px-2.5 py-1 text-xs transition ${
              candidate.label === item
                ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"
            }`}
          >
            {item}
          </button>
        ))}
      </div>
    </>
  );
}

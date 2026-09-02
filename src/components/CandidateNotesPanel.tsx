import { useState } from "react";

import { Badge, Button, Field, Note } from "@/components/ui";
import { engine, type Candidate, type VoteTally } from "@/lib/engine";

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
  const [reviewerId, setReviewerId] = useState("");
  const [tally, setTally] = useState<VoteTally | null>(null);
  const [voteBusy, setVoteBusy] = useState(false);
  const [voteError, setVoteError] = useState<string | null>(null);
  const [promotion, setPromotion] = useState<string | null>(null);

  async function loadVotes() {
    setVoteError(null);
    try {
      const result = await engine.labelVotes(candidate.candidate_id, projectId);
      setTally(result.tally);
    } catch (err) {
      setVoteError(String(err));
    }
  }

  async function castVote(value: string) {
    setVoteBusy(true);
    setVoteError(null);
    try {
      await engine.castLabelVote(candidate.candidate_id, reviewerId, value, "", projectId);
      await loadVotes();
    } catch (err) {
      setVoteError(String(err));
    } finally {
      setVoteBusy(false);
    }
  }

  async function promoteConsensus() {
    setVoteBusy(true);
    setVoteError(null);
    setPromotion(null);
    try {
      const result = await engine.promoteVoteConsensus(candidate.candidate_id, projectId);
      if (result.promoted && result.label) {
        setPromotion(`Promoted to "${result.label}" (${result.votes} votes, ${
          Math.round((result.agreement_fraction ?? 0) * 100)}% agreement).`);
        onUpdated({
          ...candidate, label: result.label,
          review: { label: result.label, note: "promoted from community votes",
                   recorded_utc: new Date().toISOString() },
        });
      } else {
        setPromotion(result.reason ?? "Not enough votes/agreement to promote yet.");
      }
    } catch (err) {
      setVoteError(String(err));
    } finally {
      setVoteBusy(false);
    }
  }

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
            className="mt-1 min-h-9 w-full resize-y rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2.5 py-1.5 text-sm text-[var(--color-text)] outline-none transition-colors focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"
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
            className={`rounded-full border px-2.5 py-1 text-xs transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] ${
              candidate.label === item
                ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"
            }`}
          >
            {item}
          </button>
        ))}
      </div>

      <div className="rounded border border-[var(--color-edge)] p-2">
        <p className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">
          Community votes (multiple independent reviewers)
        </p>
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Reviewer ID" value={reviewerId} onChange={setReviewerId}
                width="w-32" placeholder="e.g. your name or handle" />
          <Button disabled={voteBusy} onClick={() => void loadVotes()}>
            {tally ? "Refresh votes" : "Load votes"}
          </Button>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {LABELS.map((item) => (
            <button
              type="button"
              key={`vote-${item}`}
              disabled={voteBusy || !reviewerId.trim()}
              onClick={() => void castVote(item)}
              className="rounded-full border border-[var(--color-edge)] px-2.5 py-1 text-xs text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] disabled:opacity-50"
            >
              Vote {item}
            </button>
          ))}
        </div>
        {voteError && <Note tone="bad">{voteError}</Note>}
        {tally && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Badge tone="muted">{tally.total} votes</Badge>
            {Object.entries(tally.by_label).filter(([, count]) => count > 0).map(([name, count]) => (
              <Badge key={name} tone={name === tally.majority_label ? "accent" : "muted"}>
                {name.replace(/_/g, " ")}: {count}
              </Badge>
            ))}
            <Button disabled={voteBusy || tally.total === 0} onClick={() => void promoteConsensus()}>
              Promote consensus
            </Button>
          </div>
        )}
        {promotion && <Note tone="muted">{promotion}</Note>}
      </div>
    </>
  );
}

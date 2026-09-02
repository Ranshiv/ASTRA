/** Plan section 17's explainability, which had no dedicated surface.
 *
 * Every piece of this was already computed and already shipped inside each
 * candidate record; none of it was rendered. The artifact assessment in
 * particular reached the UI as a likelihood and a verdict, while the
 * `indicators` and `clearing_evidence` that justify them went unread.
 *
 * Two things are deliberately shown rather than summarised away. A score
 * component is displayed next to its weighted contribution, because 0.9 at
 * weight 0.05 and 0.5 at weight 0.25 rank the opposite way from how the raw
 * scores read. And `weight_used` is always visible: the total renormalises
 * over whatever evidence existed, so the same 0.62 means different things at
 * full weight and at half.
 */
import { useState } from "react";
import { BookOpen, Clock3, ListRestart, Orbit, Radio, ScrollText, ShieldAlert, Thermometer } from "lucide-react";

import { engine, type ReviewSelection } from "@/lib/engine";
import { Button, Empty, KeyValue, Note, Panel, useAsync } from "@/components/ui";
import { ScoreDrivers } from "./ScoreDrivers";
import { FeatureAttribution } from "./FeatureAttribution";
import { ArtifactCase } from "./ArtifactCase";
import { GwCoincidence } from "./GwCoincidence";
import { FrbCoincidence } from "./FrbCoincidence";
import { SignificanceContext } from "./SignificanceContext";
import { FollowupDraft } from "./FollowupDraft";
import { LiteratureContext } from "./LiteratureContext";
import { PhysicalContext } from "./PhysicalContext";

export function ExplainPanel({ projectId }: { projectId?: string }) {
  const candidates = useAsync(() => engine.candidates("default", 50, projectId), [projectId]);
  const [selected, setSelected] = useState<string | null>(null);
  const [reviewQueue, setReviewQueue] = useState<ReviewSelection[] | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [activeLearning, setActiveLearning] = useState(false);

  const rows = candidates.data?.candidates ?? [];
  const candidate = rows.find((row) => row.candidate_id === selected) ?? rows[0] ?? null;

  async function loadReviewQueue() {
    setReviewBusy(true);
    try {
      setReviewQueue(await engine.reviewNext("default", 12, projectId, activeLearning));
    } finally {
      setReviewBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={ScrollText}
        title="Why this candidate scored"
        description={
          candidates.loading
            ? "Loading candidates…"
            : `${rows.length} ranked candidate${rows.length === 1 ? "" : "s"}`
        }
        actions={
          <>
            <button
              type="button"
              aria-pressed={activeLearning}
              title="Reweight the review queue by learned label history instead of the fixed uncertainty-sampling heuristic"
              onClick={() => setActiveLearning((current) => !current)}
              className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] ${
                activeLearning
                  ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                  : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"
              }`}
            >
              Active learning
            </button>
            <Button icon={ListRestart} disabled={reviewBusy} onClick={() => void loadReviewQueue()}>
              Review next
            </Button>
            <Button onClick={() => void candidates.reload()}>Refresh</Button>
          </>
        }
      >
        {candidates.error && <Note tone="bad">{candidates.error}</Note>}
        {!candidates.loading && rows.length === 0 && (
          <Empty>
            No candidates to explain yet. Run the pipeline to rank a population first.
          </Empty>
        )}

        {rows.length > 0 && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-1.5">
              {rows.slice(0, 20).map((row) => (
                <button
                  key={row.candidate_id}
                  type="button"
                  onClick={() => setSelected(row.candidate_id)}
                  className={`rounded border px-2 py-0.5 text-[10px] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] ${
                    candidate?.candidate_id === row.candidate_id
                      ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                      : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-accent)]"
                  }`}
                >
                  #{row.rank} {row.object_id}
                </button>
              ))}
            </div>

            {reviewQueue && (
              <div className="rounded border border-[var(--color-edge)] bg-[var(--color-panel-2)] p-2">
                <p className="text-[11px] font-medium text-[var(--color-muted)]">
                  {activeLearning ? "Active-review queue (label-reweighted)" : "Active-review queue"}
                </p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {reviewQueue.map((item) => (
                    <button
                      key={item.candidate_id}
                      type="button"
                      title={item.reasons.join("; ")}
                      onClick={() => setSelected(item.candidate_id)}
                      className="rounded border border-[var(--color-accent)]/50 px-2 py-0.5 text-[10px] text-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"
                    >
                      {item.candidate_id} · {item.priority.toFixed(2)}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {candidate && (
              <>
                <KeyValue
                  rows={[
                    ["Object", `${candidate.survey} · ${candidate.object_id} · ${candidate.band}`],
                    ["What happened", candidate.explanation.what_happened ?? "—"],
                    ["Coverage", candidate.explanation.coverage
                      ? `${candidate.explanation.coverage.tier} · ${candidate.explanation.coverage.status}`
                      : "—"],
                  ]}
                />
                <ScoreDrivers candidate={candidate} />
                <FeatureAttribution candidate={candidate} projectId={projectId} />
              </>
            )}
          </div>
        )}
      </Panel>

      {candidate && (
        <Panel
          icon={Clock3}
          title="Follow-up visibility"
          description="Generate a local observability draft for this candidate."
        >
          <FollowupDraft candidate={candidate} projectId={projectId} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={ScrollText}
          title="Scientific significance"
          description="Calibrated tail probability, false-discovery context, and evidence completeness."
        >
          <SignificanceContext candidate={candidate} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={BookOpen}
          title="Literature context"
          description="Search cached ADS/arXiv context without turning a missing result into a novelty claim."
        >
          <LiteratureContext candidate={candidate} projectId={projectId} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={Thermometer}
          title="Physical characterization"
          description="Broadband colors and a bounded blackbody diagnostic from available photometry."
        >
          <PhysicalContext candidate={candidate} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={ShieldAlert}
          title="Artifact assessment"
          description="The instrumental case against this signal, and what clears it."
        >
          <ArtifactCase candidate={candidate} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={Orbit}
          title="Gravitational-wave coincidence"
          description="Temporal and spatial coincidence with published GW events."
        >
          <GwCoincidence candidate={candidate} />
        </Panel>
      )}

      {candidate && (
        <Panel
          icon={Radio}
          title="Fast radio burst coincidence"
          description="Temporal and spatial coincidence with published CHIME/FRB bursts."
        >
          <FrbCoincidence candidate={candidate} />
        </Panel>
      )}
    </div>
  );
}

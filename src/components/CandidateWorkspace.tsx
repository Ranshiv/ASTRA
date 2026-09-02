import { Database, Download, FlaskConical, ListChecks, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { AladinSky } from "@/components/AladinSky";
import { CandidateDiscardPilePanel } from "@/components/CandidateDiscardPilePanel";
import { CandidateFitsPanel } from "@/components/CandidateFitsPanel";
import { CandidateFoldPanel } from "@/components/CandidateFoldPanel";
import { CandidateNotesPanel } from "@/components/CandidateNotesPanel";
import { CandidateTessPanel } from "@/components/CandidateTessPanel";
import { CandidateTimelinePanel } from "@/components/CandidateTimelinePanel";
import { engine, type Candidate } from "@/lib/engine";
import { Button, Empty, Field, Note, Panel } from "@/components/ui";

const EXPORT_FORMATS = ["csv", "fits", "pdf"] as const;

export function CandidateWorkspace({ projectId }: { projectId?: string }) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [status, setStatus] = useState("No candidate run loaded.");
  const [busy, setBusy] = useState(false);
  const [exported, setExported] = useState("");
  const [candidateQuery, setCandidateQuery] = useState("");
  const [topCount, setTopCount] = useState("200");

  async function refresh() {
    try {
      const result = await engine.candidates("default", Number(topCount) || 200, projectId);
      setCandidates(result.candidates);
      setStatus(`${result.count} candidates`);
    } catch {
      setStatus("Generate a candidate run after acquiring data.");
    }
  }

  useEffect(() => {
    void refresh();
  }, [projectId]);

  const visibleCandidates = candidates.filter((candidate) =>
    `${candidate.candidate_id} ${candidate.object_id} ${candidate.survey} ${candidate.band}`
      .toLowerCase()
      .includes(candidateQuery.trim().toLowerCase()),
  );

  async function generate() {
    setBusy(true);
    setStatus("Building strata and ranking candidates…");
    try {
      const result = await engine.pipeline("default", Number(topCount) || 200, projectId);
      // The pipeline response's own `candidates` array is a separate,
      // fixed 25-row preview slice (rpc.py's `_handle_pipeline`, `preview`
      // param, never overridden here) -- unrelated to `top`/candidates_
      // built. Using it directly showed at most 25 candidates in the list
      // no matter how many were actually written. Reloading via
      // `candidates.load` (refresh, already used for the initial mount)
      // fetches up to the same `topCount` the user just built.
      await refresh();
      setStatus(`${result.candidates_built} candidates written`);
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function enrichCatalogs() {
    setBusy(true);
    setStatus("Cross-referencing SIMBAD, VSX, and configured TNS…");
    try {
      const result = await engine.catalogEnrich("default", false, false, projectId);
      await refresh();
      setStatus(`${result.candidates} candidates enriched; cached results remain available offline.`);
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function trainRanker() {
    setBusy(true);
    setStatus("Checking labels and fitting the calibrated grouped ranker…");
    try {
      const result = await engine.rankerTrain();
      if (result.ready) {
        await engine.rankerApply();
        await refresh();
        setStatus(`Ranker saved and applied: ${result.model_name}`);
      } else {
        setStatus(result.reason ?? "Ranker gate not met.");
      }
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function open(candidate: Candidate) {
    setSelected(candidate);
    try {
      const detail = await engine.candidate(candidate.candidate_id, "default", projectId);
      setSelected(detail);
    } catch {
      /* list data remains usable */
    }
  }

  function onCandidateUpdated(updated: Candidate) {
    setSelected(updated);
    setCandidates((all) => all.map((item) => (item.candidate_id === updated.candidate_id ? { ...item, label: updated.label } : item)));
  }

  async function exportRun(format: "csv" | "fits" | "pdf") {
    try {
      const result = await engine.exportCandidates(format, "default", projectId);
      setExported(result.path);
    } catch (err) {
      setExported(String(err));
    }
  }

  return (
    <Panel
      icon={ListChecks}
      title="Candidate workspace"
      description={status}
      actions={
        <>
          <Field
            id="candidate-top-count"
            label="Max candidates"
            value={topCount}
            onChange={setTopCount}
            width="w-24"
            min={1}
          />
          <Button onClick={generate} disabled={busy} loading={busy} icon={FlaskConical} tone="accent">
            Build candidates
          </Button>
          <Button onClick={() => void enrichCatalogs()} disabled={busy || candidates.length === 0} icon={Database}>
            Enrich catalogs
          </Button>
          <Button onClick={() => void trainRanker()} disabled={busy || candidates.length === 0} icon={Sparkles}>
            Train ranker
          </Button>
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-1.5">
        {EXPORT_FORMATS.map((format) => (
          <button
            type="button"
            key={format}
            onClick={() => void exportRun(format)}
            className="flex items-center gap-1 rounded-full border border-[var(--color-edge)] px-2.5 py-1 text-xs uppercase text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"
          >
            <Download size={11} strokeWidth={2} />
            {format}
          </button>
        ))}
      </div>
      {exported && (
        <div className="mt-1">
          <Note>{exported}</Note>
        </div>
      )}

      {candidates.length === 0 && (
        <div className="mt-4">
          <Empty>No candidates yet. Build candidates after acquiring data.</Empty>
        </div>
      )}

      {candidates.length > 0 && (
        <div className="mt-4 grid gap-3 md:grid-cols-[18rem_1fr] md:items-start">
          <div className="overflow-y-auto rounded border border-[var(--color-edge)] md:max-h-[70vh]">
            <div className="m-2">
              <Field
                id="candidate-filter"
                label="Filter candidates"
                value={candidateQuery}
                onChange={setCandidateQuery}
                placeholder="Filter candidates…"
                width="w-full"
              />
              <span className="sr-only">{`${visibleCandidates.length} of ${candidates.length} candidates shown`}</span>
            </div>
            {visibleCandidates.map((candidate, index) => {
              const active = selected?.candidate_id === candidate.candidate_id;
              return (
                <button
                  type="button"
                  key={candidate.candidate_id}
                  onClick={() => void open(candidate)}
                  aria-current={active ? "true" : undefined}
                  className={`block w-full border-b border-[var(--color-edge)]/50 px-2.5 py-2 text-left text-xs transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-accent)] ${
                    active ? "bg-[var(--color-accent)]/10" : index % 2 === 1 ? "bg-[var(--color-panel-2)]/40" : ""
                  }`}
                >
                  <span className="font-mono text-[var(--color-muted)]">#{candidate.rank}</span>{" "}
                  <span className="break-all">{candidate.candidate_id}</span>
                  <span className="float-right">{(candidate.score.supervised_probability ?? candidate.score.total).toFixed(3)}</span>
                  <span className="block text-[var(--color-muted)]">
                    {candidate.survey} · {candidate.band} ·{" "}
                    {candidate.score.ranking_method ? "calibrated" : candidate.explanation.coverage?.status ?? "full"}
                  </span>
                </button>
              );
            })}
          </div>

          {selected && (
            <div className="space-y-3 md:max-h-[70vh] md:overflow-y-auto md:pr-1" key={selected.candidate_id}>
              <div>
                <h3 className="font-mono text-xs">{selected.candidate_id}</h3>
                <p className="text-xs text-[var(--color-muted)]">{selected.explanation.what_happened}</p>
              </div>

              {selected.path && (
                <CandidateFoldPanel path={selected.path} bestPeriodDays={selected.features.best_period_days as number | null} />
              )}
              <AladinSky ra={selected.ra_deg} dec={selected.dec_deg} />

              <div className="grid gap-3 lg:grid-cols-2">
                <div className="rounded border border-[var(--color-edge)] p-2">
                  <p className="text-xs font-medium">Why it was flagged</p>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted)]">
                    {(selected.explanation.why_flagged ?? ["No component could be computed."]).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                  {selected.explanation.recommended_actions && (
                    <>
                      <p className="mt-2 text-xs font-medium">Recommended actions</p>
                      <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted)]">
                        {selected.explanation.recommended_actions.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </>
                  )}
                </div>
                <div className="rounded border border-[var(--color-edge)] p-2">
                  <p className="text-xs font-medium">Evidence and artifact assessment</p>
                  <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                    <dt className="text-[var(--color-muted)]">Score</dt>
                    <dd>{selected.score.total.toFixed(3)}</dd>
                    <dt className="text-[var(--color-muted)]">Artifact likelihood</dt>
                    <dd>{(selected.artifact.likelihood ?? 0).toFixed(3)}</dd>
                    <dt className="text-[var(--color-muted)]">Verdict</dt>
                    <dd>{selected.artifact.verdict ?? "not assessed"}</dd>
                    <dt className="text-[var(--color-muted)]">Resolved surveys</dt>
                    <dd>{selected.explanation.supporting_observations?.surveys_resolving ?? "—"}</dd>
                  </dl>
                </div>
              </div>

              <CandidateTimelinePanel candidateId={selected.candidate_id} projectId={projectId} />

              <CandidateNotesPanel candidate={selected} projectId={projectId} onUpdated={onCandidateUpdated} />

              <CandidateFitsPanel candidate={selected} projectId={projectId} />

              <CandidateTessPanel candidate={selected} />

              <CandidateDiscardPilePanel candidate={selected} />
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

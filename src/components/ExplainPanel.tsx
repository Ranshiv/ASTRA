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

import { engine, type Candidate, type PhysicalCharacterization, type ReviewSelection } from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Table, num, useAsync } from "@/components/ui";

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

function SignificanceContext({ candidate }: { candidate: Candidate }) {
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

function FollowupDraft({ candidate }: { candidate: Candidate }) {
  const [latitude, setLatitude] = useState("43.65");
  const [longitude, setLongitude] = useState("-79.38");
  const [duration, setDuration] = useState("12");
  const [minimumAltitude, setMinimumAltitude] = useState("30");
  const [plan, setPlan] = useState<Awaited<ReturnType<typeof engine.followupPlan>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function makePlan() {
    setBusy(true);
    setError(null);
    try {
      setPlan(await engine.followupPlan({
        raDeg: candidate.ra_deg,
        decDeg: candidate.dec_deg,
        latitudeDeg: Number(latitude),
        longitudeDeg: Number(longitude),
        durationHours: Number(duration),
        minAltitudeDeg: Number(minimumAltitude),
        targetId: candidate.candidate_id,
      }));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Latitude" value={latitude} onChange={setLatitude} width="w-24" />
        <Field label="Longitude" value={longitude} onChange={setLongitude} width="w-24" />
        <Field label="Hours" value={duration} onChange={setDuration} width="w-16" />
        <Field label="Min altitude" value={minimumAltitude} onChange={setMinimumAltitude} width="w-20" />
        <Button icon={Clock3} tone="accent" disabled={busy} onClick={() => void makePlan()}>
          {busy ? "Planning…" : "Plan visibility"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {!plan && !error && <Note>Coordinates are pre-filled from the candidate. The result is a draft geometry check, not an observing request.</Note>}
      {plan && (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={plan.visible ? "ok" : "warn"}>{plan.visible ? "visible window found" : "not visible"}</Badge>
            <Badge tone="muted">{plan.windows.length} window{plan.windows.length === 1 ? "" : "s"}</Badge>
            {plan.best_slot && <Badge tone="accent">best airmass {plan.best_slot.airmass.toFixed(2)}</Badge>}
          </div>
          {plan.best_slot && (
            <KeyValue rows={[
              ["Best UTC", plan.best_slot.utc],
              ["Altitude", `${plan.best_slot.altitude_deg.toFixed(1)}°`],
              ["Azimuth", `${plan.best_slot.azimuth_deg.toFixed(1)}°`],
            ]} />
          )}
          {plan.windows.length > 0 && (
            <Table head={["Start UTC", "End UTC", "Slots"]}>
              {plan.windows.map((window) => (
                <tr key={`${window.start_utc}-${window.end_utc}`} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono text-[11px]">{window.start_utc}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px]">{window.end_utc}</td>
                  <td className="px-2 py-1.5">{window.slots}</td>
                </tr>
              ))}
            </Table>
          )}
          <Note tone="warn">{plan.caveats[0]} No observation request was submitted.</Note>
        </>
      )}
    </div>
  );
}

function LiteratureContext({ candidate, projectId }: { candidate: Candidate; projectId?: string }) {
  const [result, setResult] = useState(candidate.literature ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function search() {
    setBusy(true);
    setError(null);
    try {
      const response = await engine.literatureSearch({
        objectId: candidate.object_id,
        terms: candidate.explanation.resembles ?? [],
        eventIds: candidate.event_ids ?? [],
        projectId,
      });
      setResult(response);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const records = result?.records ?? [];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result?.complete ? "ok" : "muted"}>
          {result ? `${records.length} record${records.length === 1 ? "" : "s"}` : "not queried"}
        </Badge>
        {result && <Badge tone="muted">{result.complete ? "providers complete" : "partial / unavailable"}</Badge>}
        <Button icon={BookOpen} disabled={busy} onClick={() => void search()}>
          {busy ? "Searching…" : "Search literature"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {records.length === 0 && !error && (
        <Empty>{result ? "No matching records were returned; this is not a novelty claim." : "No literature search has been run for this candidate."}</Empty>
      )}
      {records.length > 0 && (
        <Table head={["Year", "Title", "Provider"]}>
          {records.slice(0, 8).map((record, index) => (
            <tr key={`${record.url ?? record.title ?? "record"}-${index}`} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{record.year ?? "—"}</td>
              <td className="px-2 py-1.5">
                {record.url ? <a href={record.url} target="_blank" rel="noreferrer" className="text-[var(--color-accent)]">{record.title ?? record.url}</a> : record.title ?? "—"}
              </td>
              <td className="px-2 py-1.5 text-[var(--color-muted)]">{record.provider}</td>
            </tr>
          ))}
        </Table>
      )}
      <Note tone="warn">Literature provides context and provenance only; it never changes the ASTRA score or treats an unavailable provider as a no-match.</Note>
    </div>
  );
}

function PhysicalContext({ candidate }: { candidate: Candidate }) {
  const [physical, setPhysical] = useState<PhysicalCharacterization | null>(
    candidate.physical_characterization as PhysicalCharacterization | undefined ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function characterize() {
    setBusy(true);
    setError(null);
    try {
      setPhysical(await engine.physicalCharacterize(candidate.features));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }
  if (!physical) {
    return (
      <div className="flex flex-col gap-2">
        <Empty>No broadband physical characterization has been run for this candidate.</Empty>
        <Button icon={Thermometer} disabled={busy} onClick={() => void characterize()}>
          {busy ? "Characterizing…" : "Characterize available photometry"}
        </Button>
        {error && <Note tone="bad">{error}</Note>}
      </div>
    );
  }
  const temperature = physical.temperature_k;
  const bands = Array.isArray(physical.bands_used) ? physical.bands_used.join(", ") : "—";
  const colors = physical.colors && typeof physical.colors === "object"
    ? physical.colors as Record<string, unknown> : null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={physical.quality === "usable" ? "ok" : "warn"}>{String(physical.quality ?? "unknown")}</Badge>
        <Badge tone="muted">{temperature == null ? "temperature unavailable" : `${Number(temperature).toFixed(0)} K`}</Badge>
        <Badge tone="muted">bands: {bands}</Badge>
        <Button icon={Thermometer} disabled={busy} onClick={() => void characterize()}>
          {busy ? "Updating…" : "Recalculate"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {colors && <KeyValue rows={Object.entries(colors).map(([key, value]) => [key, String(value)] as [string, string])} />}
      {Array.isArray(physical.warnings) && physical.warnings.length > 0 && (
        <ul className="list-inside list-disc text-xs text-[var(--color-muted)]">
          {physical.warnings.map((warning, index) => <li key={index}>{String(warning)}</li>)}
        </ul>
      )}
      <Note tone="warn">This is a bounded broadband SED diagnostic, not a spectral classification or extinction measurement.</Note>
    </div>
  );
}

export function ExplainPanel({ projectId }: { projectId?: string }) {
  const candidates = useAsync(() => engine.candidates("default", 50, projectId), [projectId]);
  const [selected, setSelected] = useState<string | null>(null);
  const [reviewQueue, setReviewQueue] = useState<ReviewSelection[] | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);

  const rows = candidates.data?.candidates ?? [];
  const candidate = rows.find((row) => row.candidate_id === selected) ?? rows[0] ?? null;

  async function loadReviewQueue() {
    setReviewBusy(true);
    try {
      setReviewQueue(await engine.reviewNext("default", 12, projectId));
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
                  className={`rounded border px-2 py-0.5 text-[10px] transition ${
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
                <p className="text-[11px] font-medium text-[var(--color-muted)]">Active-review queue</p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {reviewQueue.map((item) => (
                    <button
                      key={item.candidate_id}
                      type="button"
                      title={item.reasons.join("; ")}
                      onClick={() => setSelected(item.candidate_id)}
                      className="rounded border border-[var(--color-accent)]/50 px-2 py-0.5 text-[10px] text-[var(--color-accent)]"
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
          <FollowupDraft candidate={candidate} />
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

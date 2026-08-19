import {
  ChevronRight,
  Database,
  Download,
  FlaskConical,
  Image as ImageIcon,
  ListChecks,
  RefreshCcw,
  RotateCcw,
  Satellite,
  Sparkles,
} from "lucide-react";
import { useEffect, useState } from "react";

import { AladinSky } from "@/components/AladinSky";
import { FitsViewer } from "@/components/FitsViewer";
import { LightCurvePlot } from "@/components/LightCurvePlot";
import { SectionHeader } from "@/components/SectionHeader";
import {
  engine,
  type Candidate,
  type CurvePayload,
  type FoldedCurve,
  type ImageProduct,
  type TessPhotometryPayload,
  type TessTpfProduct,
  type ZtfImageMetadata,
  type CandidateTimeline,
  type CandidateTimelineCurve,
  type ImageFeaturePayload,
} from "@/lib/engine";

const LABELS = ["interesting", "artifact", "known_object", "uncertain", "needs_follow_up"] as const;
const EXPORT_FORMATS = ["csv", "fits", "pdf"] as const;

function timelineCurvePayload(curve: CandidateTimelineCurve): CurvePayload {
  const finite = curve.values.filter(Number.isFinite);
  const mean = finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
  const variance = finite.length
    ? finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length
    : 0;
  return {
    path: curve.path,
    survey: curve.survey,
    release: curve.release,
    object_id: curve.object_id,
    band: curve.band,
    value_kind: curve.value_kind,
    time_system: curve.time_system,
    points: curve.points,
    time_span_days: curve.time_end - curve.time_start,
    mean_value: mean,
    std_value: Math.sqrt(variance),
    time: curve.times,
    value: curve.values,
    value_err: curve.values.map(() => Number.NaN),
    shown_points: curve.times.length,
    downsampled: curve.times.length < curve.points,
  };
}

function asTessCurve(payload: TessPhotometryPayload): CurvePayload {
  const finite = payload.flux.filter(Number.isFinite);
  const mean = finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
  const variance = finite.length
    ? finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length
    : 0;
  return {
    path: payload.curve_path ?? payload.path,
    survey: "TESS",
    release: `tpf-s${payload.sector ?? "unknown"}`,
    object_id: payload.source ?? "TPF",
    band: "TESS",
    value_kind: "flux",
    time_system: "BJD_TDB",
    points: payload.points,
    time_span_days: payload.time.length > 1 ? payload.time[payload.time.length - 1] - payload.time[0] : 0,
    mean_value: mean,
    std_value: Math.sqrt(variance),
    time: payload.time,
    value: payload.flux,
    value_err: payload.flux_err.map((value) => value ?? Number.NaN),
    shown_points: payload.shown_points,
    downsampled: payload.downsampled,
  };
}

export function CandidateWorkspace({ projectId }: { projectId?: string }) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [curve, setCurve] = useState<CurvePayload | null>(null);
  const [folded, setFolded] = useState<FoldedCurve | null>(null);
  const [period, setPeriod] = useState("");
  const [foldError, setFoldError] = useState<string | null>(null);
  const [status, setStatus] = useState("No candidate run loaded.");
  const [busy, setBusy] = useState(false);
  const [fitsPath, setFitsPath] = useState("");
  const [cutouts, setCutouts] = useState<ZtfImageMetadata[]>([]);
  const [products, setProducts] = useState<ImageProduct[]>([]);
  const [cutoutStatus, setCutoutStatus] = useState("");
  const [imageFeatures, setImageFeatures] = useState<ImageFeaturePayload | null>(null);
  const [exported, setExported] = useState("");
  const [tessSector, setTessSector] = useState("1");
  const [tessProduct, setTessProduct] = useState<TessTpfProduct | null>(null);
  const [tessPhotometry, setTessPhotometry] = useState<TessPhotometryPayload | null>(null);
  const [tessStatus, setTessStatus] = useState("");
  const [timeline, setTimeline] = useState<CandidateTimeline | null>(null);
  const [note, setNote] = useState("");

  async function refresh() {
    try {
      const result = await engine.candidates("default", 50, projectId);
      setCandidates(result.candidates);
      setStatus(`${result.count} candidates`);
    } catch {
      setStatus("Generate a candidate run after acquiring data.");
    }
  }

  useEffect(() => {
    void refresh();
  }, [projectId]);

  async function generate() {
    setBusy(true);
    setStatus("Building strata and ranking candidates…");
    try {
      const result = await engine.pipeline("default", 200, projectId);
      setCandidates(result.candidates);
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
    setCurve(null);
    setFolded(null);
    setPeriod("");
    setFoldError(null);
    setFitsPath("");
    setCutouts([]);
    setProducts([]);
    setCutoutStatus("");
    setImageFeatures(null);
    setTessProduct(null);
    setTessPhotometry(null);
    setTessStatus("");
    setTimeline(null);
    setNote("");
    if (candidate.path) {
      try {
        setCurve(await engine.curveGet(candidate.path, 2000, "BJD_TDB"));
      } catch {
        /* detail remains usable */
      }
      try {
        const detail = await engine.candidate(candidate.candidate_id, "default", projectId);
        setNote(detail.review?.note ?? "");
        setSelected(detail);
        const bestPeriod = detail.features.best_period_days;
        if (typeof bestPeriod === "number" && Number.isFinite(bestPeriod) && bestPeriod > 0) {
          setPeriod(String(bestPeriod));
        }
      } catch {
        /* list data remains usable */
      }
      try {
        setTimeline(await engine.candidateTimeline(candidate.candidate_id, "default", projectId));
      } catch {
        /* timeline is an enhancement; preserve the curve view */
      }
    }
  }

  async function applyFold() {
    if (!curve) return;
    const days = Number(period);
    if (!Number.isFinite(days) || days <= 0) {
      setFoldError("Enter a positive period in days.");
      return;
    }
    setFoldError(null);
    try {
      setFolded(await engine.curveFold(curve.path, days));
    } catch (err) {
      setFoldError(String(err));
    }
  }

  async function findCutouts() {
    if (!selected) return;
    setCutoutStatus("Searching ZTF image metadata…");
    try {
      const found = await engine.ztfImageSearch(selected.ra_deg, selected.dec_deg);
      setCutouts(found);
      setCutoutStatus(
        found.length ? `${found.length} archive image products found.` : "No overlapping ZTF image products found.",
      );
    } catch (err) {
      setCutoutStatus(String(err));
    }
  }

  async function downloadCutout(metadata: ZtfImageMetadata) {
    if (!selected) return;
    setCutoutStatus("Queueing a quota-safe FITS cutout download…");
    try {
      const job = await engine.ztfImageDownload({ raDeg: selected.ra_deg, decDeg: selected.dec_deg, metadata });
      let status = await engine.jobStatus(job.job_id);
      while (["queued", "running", "retrying"].includes(status.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 750));
        status = await engine.jobStatus(job.job_id);
      }
      if (status.status !== "completed") throw new Error(status.error ?? `download ${status.status}`);
      const product = status.result as ImageProduct;
      setProducts((current) => [product, ...current.filter((item) => item.product_id !== product.product_id)]);
      setFitsPath(product.path);
      setCutoutStatus(
        product.reused ? "Using the existing verified FITS cutout." : "FITS cutout downloaded, checksummed, and validated.",
      );
    } catch (err) {
      setCutoutStatus(String(err));
    }
  }

  async function downloadTessTpf() {
    if (!selected) return;
    const sector = Number(tessSector);
    if (!Number.isInteger(sector) || sector < 1) {
      setTessStatus("Enter a valid TESS sector number.");
      return;
    }
    setTessStatus("Queueing an explicit, quota-safe TESS target-pixel download…");
    try {
      const job = await engine.tessTpfDownload({
        raDeg: selected.ra_deg,
        decDeg: selected.dec_deg,
        sector,
        targetId: selected.object_id || selected.candidate_id,
      });
      let status = await engine.jobStatus(job.job_id);
      while (["queued", "running", "retrying"].includes(status.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 750));
        status = await engine.jobStatus(job.job_id);
      }
      if (status.status !== "completed") throw new Error(status.error ?? `TPF download ${status.status}`);
      const product = status.result as TessTpfProduct;
      setTessProduct(product);
      setTessStatus(
        product.reused
          ? "Using the existing verified TESS target-pixel file."
          : "TESS target-pixel file downloaded, checksummed, and validated.",
      );
    } catch (err) {
      setTessStatus(String(err));
    }
  }

  async function extractTessPhotometry() {
    if (!selected || !tessProduct) return;
    setTessStatus("Extracting background-subtracted aperture photometry…");
    try {
      const payload = await engine.tessTpfPhotometry({
        path: tessProduct.path,
        raDeg: selected.ra_deg,
        decDeg: selected.dec_deg,
        targetId: selected.object_id || selected.candidate_id,
        persist: true,
      });
      setTessPhotometry(payload);
      setTessStatus(`${payload.points.toLocaleString()} quality-filtered cadences persisted. Blend risk: ${payload.blend.risk}.`);
    } catch (err) {
      setTessStatus(String(err));
    }
  }

  async function saveNote() {
    if (!selected) return;
    try {
      const labelValue = selected.label ?? "uncertain";
      await engine.label(selected.candidate_id, labelValue, note, projectId);
      setSelected({ ...selected, review: { label: labelValue, note, recorded_utc: new Date().toISOString() } });
      setStatus("Research note saved.");
    } catch (err) {
      setStatus(String(err));
    }
  }

  async function label(label: string) {
    if (!selected) return;
    await engine.label(selected.candidate_id, label, note, projectId);
    setSelected({ ...selected, label, review: { label, note, recorded_utc: new Date().toISOString() } });
    setCandidates((all) => all.map((item) => (item.candidate_id === selected.candidate_id ? { ...item, label } : item)));
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
    <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
      <SectionHeader
        icon={ListChecks}
        title="Candidate workspace"
        description={status}
        actions={
          <>
            <button
              type="button"
              onClick={generate}
              disabled={busy}
              className="flex items-center gap-1.5 rounded border border-[var(--color-accent)] px-2.5 py-1 text-xs text-[var(--color-accent)] disabled:opacity-40"
            >
              <FlaskConical size={12} strokeWidth={2} />
              {busy ? "Working…" : "Build candidates"}
            </button>
            <button
              type="button"
              onClick={() => void enrichCatalogs()}
              disabled={busy || candidates.length === 0}
              className="flex items-center gap-1.5 rounded border border-[var(--color-edge)] px-2.5 py-1 text-xs disabled:opacity-40"
            >
              <Database size={12} strokeWidth={2} />
              Enrich catalogs
            </button>
            <button
              type="button"
              onClick={() => void trainRanker()}
              disabled={busy || candidates.length === 0}
              className="flex items-center gap-1.5 rounded border border-[var(--color-edge)] px-2.5 py-1 text-xs disabled:opacity-40"
            >
              <Sparkles size={12} strokeWidth={2} />
              Train ranker
            </button>
          </>
        }
      />

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        {EXPORT_FORMATS.map((format) => (
          <button
            type="button"
            key={format}
            onClick={() => void exportRun(format)}
            className="flex items-center gap-1 rounded-full border border-[var(--color-edge)] px-2.5 py-1 text-xs uppercase text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
          >
            <Download size={11} strokeWidth={2} />
            {format}
          </button>
        ))}
      </div>
      {exported && <p className="mt-1 break-all font-mono text-xs text-[var(--color-muted)]">{exported}</p>}

      {candidates.length > 0 && (
        <div className="mt-4 grid gap-3 md:grid-cols-[18rem_1fr] md:items-start">
          <div className="overflow-y-auto rounded border border-[var(--color-edge)] md:max-h-[70vh]">
            {candidates.map((candidate, index) => {
              const active = selected?.candidate_id === candidate.candidate_id;
              return (
                <button
                  type="button"
                  key={candidate.candidate_id}
                  onClick={() => void open(candidate)}
                  className={`block w-full border-b border-[var(--color-edge)]/50 px-2.5 py-2 text-left text-xs transition ${
                    active ? "bg-[var(--color-accent)]/10" : index % 2 === 1 ? "bg-[var(--color-panel-2)]/40" : ""
                  }`}
                >
                  <span className="font-mono text-[var(--color-muted)]">#{candidate.rank}</span> {candidate.candidate_id}
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
            <div className="space-y-3 md:max-h-[70vh] md:overflow-y-auto md:pr-1">
              <div>
                <h3 className="font-mono text-xs">{selected.candidate_id}</h3>
                <p className="text-xs text-[var(--color-muted)]">{selected.explanation.what_happened}</p>
              </div>

              {curve && (
                <div>
                  <LightCurvePlot curve={curve} folded={folded} />
                  <div className="mt-1 flex items-center gap-2">
                    <input
                      value={period}
                      onChange={(event) => setPeriod(event.target.value)}
                      placeholder="Fold period (days)"
                      inputMode="decimal"
                      className="flex-1 rounded border border-[var(--color-edge)] bg-[var(--color-void)]
                                 px-2 py-1 text-xs outline-none focus:border-[var(--color-accent)]"
                    />
                    <button
                      type="button"
                      onClick={applyFold}
                      className="flex items-center gap-1.5 rounded border border-[var(--color-edge)] px-2.5 py-1 text-xs
                                 transition hover:border-[var(--color-accent)]"
                    >
                      <RefreshCcw size={12} strokeWidth={2} />
                      Fold
                    </button>
                    {folded && (
                      <button
                        type="button"
                        onClick={() => setFolded(null)}
                        className="flex items-center gap-1.5 rounded border border-[var(--color-edge)] px-2.5 py-1 text-xs
                                   text-[var(--color-muted)] transition hover:border-[var(--color-accent)]"
                      >
                        <RotateCcw size={12} strokeWidth={2} />
                        Unfold
                      </button>
                    )}
                  </div>
                  {foldError && <p className="mt-1 text-xs text-[var(--color-bad)]">{foldError}</p>}
                </div>
              )}
              <AladinSky ra={selected.ra_deg} dec={selected.dec_deg} />

              <div className="grid gap-3 lg:grid-cols-2">
                <div className="rounded border border-[var(--color-edge)] p-2">
                  <p className="text-xs font-medium">Why it was flagged</p>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted)]">
                    {(selected.explanation.why_flagged ?? ["No component could be computed."]).map((item) => <li key={item}>{item}</li>)}
                  </ul>
                  {selected.explanation.recommended_actions && <>
                    <p className="mt-2 text-xs font-medium">Recommended actions</p>
                    <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted)]">
                      {selected.explanation.recommended_actions.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </>}
                </div>
                <div className="rounded border border-[var(--color-edge)] p-2">
                  <p className="text-xs font-medium">Evidence and artifact assessment</p>
                  <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                    <dt className="text-[var(--color-muted)]">Score</dt><dd>{selected.score.total.toFixed(3)}</dd>
                    <dt className="text-[var(--color-muted)]">Artifact likelihood</dt><dd>{(selected.artifact.likelihood ?? 0).toFixed(3)}</dd>
                    <dt className="text-[var(--color-muted)]">Verdict</dt><dd>{selected.artifact.verdict ?? "not assessed"}</dd>
                    <dt className="text-[var(--color-muted)]">Resolved surveys</dt><dd>{selected.explanation.supporting_observations?.surveys_resolving ?? "—"}</dd>
                  </dl>
                </div>
              </div>

              <details className="group rounded border border-[var(--color-edge)] p-2" open={Boolean(timeline)}>
                <summary className="cursor-pointer list-none text-xs font-medium [&::-webkit-details-marker]:hidden">Combined observation timeline</summary>
                {timeline?.warning && <p className="mt-1 text-xs text-[var(--color-warn)]">{timeline.warning}</p>}
                {timeline && timeline.events.length > 0 ? (
                  <div className="mt-2 space-y-2">
                    {timeline.events.map((event) => {
                      const span = Math.max(event.time_end - event.time_start, 1e-6);
                      return <div key={`${event.survey}-${event.release}-${event.object_id}-${event.band}`} className="text-xs">
                        <div className="flex justify-between gap-2 text-[var(--color-muted)]"><span>{event.survey} · {event.release} · {event.band}</span><span>{event.points.toLocaleString()} pts</span></div>
                        <div className="mt-1 h-2 overflow-hidden rounded bg-[var(--color-panel-2)]" title={`${event.time_start.toFixed(3)}–${event.time_end.toFixed(3)} BJD_TDB`}>
                          <div className={`h-full rounded ${event.resolved ? "bg-[var(--color-accent)]" : "bg-[var(--color-warn)]"}`} style={{ width: `${Math.max(4, Math.min(100, span / Math.max(...timeline.events.map((item) => item.time_end - item.time_start), 1e-6) * 100))}%` }} />
                        </div>
                      </div>;
                    })}
                    {timeline.curves.length > 1 && (
                      <div className="mt-3 grid gap-2 lg:grid-cols-2">
                        {timeline.curves.slice(0, 6).map((item) => (
                          <div key={`plot-${item.survey}-${item.release}-${item.object_id}-${item.band}`} className="rounded border border-[var(--color-edge)]/60 p-1">
                            <p className="px-1 py-1 text-[11px] text-[var(--color-muted)]">{item.survey} · {item.band} · {item.resolved ? "resolved" : "blended"}</p>
                            <LightCurvePlot curve={timelineCurvePayload(item)} folded={null} />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : <p className="mt-1 text-xs text-[var(--color-muted)]">No matching multi-survey curves were found.</p>}
              </details>

              <div className="rounded border border-[var(--color-edge)] p-2">
                <label className="block text-xs"><span className="text-[var(--color-muted)]">Research note</span><textarea value={note} onChange={(event) => setNote(event.target.value)} rows={2} maxLength={4000} className="mt-1 w-full resize-y rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1.5" placeholder="Record why this candidate matters or what to check next." /></label>
                <button type="button" onClick={() => void saveNote()} disabled={!selected} className="mt-2 rounded border border-[var(--color-edge)] px-2 py-1 text-xs">Save note</button>
              </div>

              {selected.explanation.coverage?.status === "insufficient_data_lt_10_points" && (
                <p className="text-xs text-[var(--color-warn)]">Retained for review: fewer than 10 finite points.</p>
              )}

              <div className="flex flex-wrap gap-1.5">
                {LABELS.map((item) => (
                  <button
                    type="button"
                    key={item}
                    onClick={() => void label(item)}
                    className={`rounded-full border px-2.5 py-1 text-xs transition ${
                      selected.label === item
                        ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                        : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>

              <details className="group rounded border border-[var(--color-edge)] p-2">
                <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs [&::-webkit-details-marker]:hidden">
                  <ChevronRight
                    size={12}
                    strokeWidth={2}
                    className="text-[var(--color-muted)] transition-transform duration-200 group-open:rotate-90"
                  />
                  <ImageIcon size={13} strokeWidth={2} />
                  FITS image
                </summary>
                <div className="my-2 flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={() => void findCutouts()}
                    className="rounded border border-[var(--color-edge)] px-2 py-1 text-xs"
                  >
                    Find ZTF cutouts
                  </button>
                  {cutouts.slice(0, 5).map((item, index) => (
                    <button
                      type="button"
                      key={`${item.filefracday}-${index}`}
                      onClick={() => void downloadCutout(item)}
                      className="rounded border border-[var(--color-edge)] px-2 py-1 text-xs"
                    >
                      Download {String(item.filtercode)}
                    </button>
                  ))}
                </div>
                {cutoutStatus && <p className="text-xs text-[var(--color-muted)]">{cutoutStatus}</p>}
                {products.map((item) => (
                  <button
                    type="button"
                    key={item.product_id}
                    onClick={() => setFitsPath(item.path)}
                    className="block max-w-full truncate text-left font-mono text-xs text-[var(--color-accent)]"
                  >
                    {item.product_id.slice(0, 12)} · {(item.bytes / 1024).toFixed(0)} KiB
                  </button>
                ))}
                <input
                  value={fitsPath}
                  onChange={(event) => setFitsPath(event.target.value)}
                  placeholder="Path to a local FITS cutout"
                  className="my-2 w-full rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1 text-xs"
                />
                {fitsPath ? <>
                  <FitsViewer path={fitsPath} />
                  <button type="button" onClick={async () => {
                    try {
                      setCutoutStatus("Extracting bounded image morphology features…");
                      const result = await engine.imageFeatures(fitsPath, projectId);
                      setImageFeatures(result);
                      setCutoutStatus(result.output_path ? `Image features persisted: ${result.output_path}` : "Image features extracted.");
                    } catch (err) {
                      setCutoutStatus(String(err));
                    }
                  }} className="mt-2 rounded border border-[var(--color-edge)] px-2 py-1 text-xs">Extract image features</button>
                  {imageFeatures && <div className="mt-2 rounded border border-[var(--color-edge)] bg-[var(--color-panel-2)]/30 p-2">
                    <p className="text-xs font-medium">Image-derived features</p>
                    <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                      {Object.entries(imageFeatures.features).map(([key, value]) => <><dt key={`${key}-label`} className="text-[var(--color-muted)]">{key.replace(/_/g, " ")}</dt><dd key={`${key}-value`} className="font-mono">{value === null ? "—" : typeof value === "number" ? value.toFixed(4) : String(value)}</dd></>)}
                    </dl>
                  </div>}
                </> : null}
              </details>

              <details className="group rounded border border-[var(--color-edge)] p-2">
                <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs [&::-webkit-details-marker]:hidden">
                  <ChevronRight
                    size={12}
                    strokeWidth={2}
                    className="text-[var(--color-muted)] transition-transform duration-200 group-open:rotate-90"
                  />
                  <Satellite size={13} strokeWidth={2} />
                  TESS target-pixel file
                </summary>
                <p className="mt-1 text-xs text-[var(--color-muted)]">
                  Candidate-scale only. TESS pixels span about 21 arcsec, so this remains neighbourhood photometry rather
                  than resolved stellar confirmation.
                </p>
                <div className="my-2 flex flex-wrap items-center gap-1.5">
                  <input
                    value={tessSector}
                    onChange={(event) => setTessSector(event.target.value)}
                    inputMode="numeric"
                    aria-label="TESS sector"
                    className="w-20 rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    onClick={() => void downloadTessTpf()}
                    className="rounded border border-[var(--color-edge)] px-2 py-1 text-xs"
                  >
                    Download TPF
                  </button>
                  {tessProduct && (
                    <button
                      type="button"
                      onClick={() => void extractTessPhotometry()}
                      className="rounded border border-[var(--color-accent)] px-2 py-1 text-xs text-[var(--color-accent)]"
                    >
                      Extract aperture curve
                    </button>
                  )}
                </div>
                {tessStatus && <p className="text-xs text-[var(--color-muted)]">{tessStatus}</p>}
                {tessProduct && <p className="mt-1 break-all font-mono text-xs text-[var(--color-accent)]">{tessProduct.path}</p>}
                {tessPhotometry && (
                  <>
                    <LightCurvePlot curve={asTessCurve(tessPhotometry)} folded={null} />
                    <p className="mt-1 text-xs text-[var(--color-warn)]">
                      Blend assessment: {tessPhotometry.blend.risk} risk; {tessPhotometry.blend.neighbors_in_aperture}{" "}
                      neighbour(s) in the aperture. {tessPhotometry.blend.warning}
                    </p>
                  </>
                )}
              </details>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

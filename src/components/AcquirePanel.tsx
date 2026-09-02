import { Play, Radar, X } from "lucide-react";
import { useEffect, useState } from "react";

import {
  engine,
  type AcquisitionResult,
  type EngineJob,
  type ProjectAcquisitionResult,
  type ProjectRegion,
  type SurveyInfo,
} from "@/lib/engine";
import { Badge, Button, Field, Note, Panel, Table } from "@/components/ui";

const ALL_SURVEYS = ["ztf", "gaia", "tess", "sdss", "panstarrs"];

export function AcquirePanel({
  surveys,
  projectId,
  queryRegions,
}: {
  surveys: SurveyInfo[];
  projectId?: string;
  queryRegions?: ProjectRegion[];
}) {
  const [ra, setRa] = useState("291.3663");
  const [dec, setDec] = useState("42.7844");
  const [radius, setRadius] = useState("10");
  const [limit, setLimit] = useState("200");
  const [selected, setSelected] = useState<string[]>(ALL_SURVEYS);
  const [job, setJob] = useState<EngineJob | null>(null);
  const [result, setResult] = useState<AcquisitionResult | ProjectAcquisitionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const toggle = (name: string) =>
    setSelected((current) =>
      current.includes(name)
        ? current.filter((s) => s !== name)
        : [...current, name],
      );

  async function run() {
    const validationError = validateForm();
    if (validationError) {
      setFormError(validationError);
      return;
    }
    setFormError(null);
    setError(null);
    setResult(null);
    try {
      const params = {
        ra_deg: Number(ra),
        dec_deg: Number(dec),
        radius_arcsec: Number(radius),
        surveys: selected,
        limit: Number(limit),
        project_id: projectId,
      };
      const key = ["acquire", params.ra_deg, params.dec_deg, params.radius_arcsec, params.limit, ...[...selected].sort()].join(":");
      const submitted = await engine.jobSubmit("acquire.cone", params, projectId, key);

      // Idempotent replay: submitting the same cone+surveys again while an
      // earlier run with the same key exists returns {existing: true} with
      // no result attached, not a fresh queued job. If that earlier run
      // already finished, its status is already terminal, so the polling
      // effect below would never fire and the result would never appear —
      // fetching the full record now (which DOES carry the result) is what
      // makes a repeat click show the answer instead of silently doing
      // nothing.
      const full = await engine.jobStatus(submitted.job_id);
      setJob(full);
      if (full.status === "completed") {
        setResult(full.result as AcquisitionResult);
      } else if (full.status === "failed") {
        setError(full.error ?? "Acquisition job failed.");
      }
    } catch (err) {
      setError(String(err));
    }
  }

  async function runProject() {
    if (!projectId || !queryRegions?.length) return;
    setError(null);
    setResult(null);
    try {
      const params = {
        surveys: selected,
        limit: Number(limit),
        project_id: projectId,
        skip_existing: true,
      };
      const key = ["acquire-project", projectId, ...[...selected].sort()].join(":");
      const submitted = await engine.jobSubmit("acquire.project", params, projectId, key);

      const full = await engine.jobStatus(submitted.job_id);
      setJob(full);
      if (full.status === "completed") {
        setResult(full.result as ProjectAcquisitionResult);
      } else if (full.status === "failed") {
        setError(full.error ?? "Acquisition job failed.");
      }
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.status)) return;

    let active = true;
    const poll = async () => {
      try {
        const next = await engine.jobStatus(job.job_id);
        if (!active) return;
        setJob(next);
        if (next.status === "completed") {
          setResult(next.result as AcquisitionResult | ProjectAcquisitionResult);
        } else if (next.status === "failed") {
          setError(next.error ?? "Acquisition job failed.");
        }
      } catch (err) {
        if (active) setError(String(err));
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 750);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const running = Boolean(job && !["completed", "failed", "cancelled"].includes(job.status));

  async function cancel() {
    if (!job || !running) return;
    try {
      setJob(await engine.jobCancel(job.job_id));
    } catch (err) {
      setError(String(err));
    }
  }

  const progress = job?.progress;

  const valid =
    Number.isFinite(Number(ra)) && Number(ra) >= 0 && Number(ra) < 360 &&
    Number.isFinite(Number(dec)) && Number(dec) >= -90 && Number(dec) <= 90 &&
    Number.isFinite(Number(radius)) && Number(radius) > 0 &&
    Number.isFinite(Number(limit)) && Number(limit) > 0 && selected.length > 0;

  function validateForm() {
    const raValue = Number(ra);
    const decValue = Number(dec);
    const radiusValue = Number(radius);
    const limitValue = Number(limit);
    if (!Number.isFinite(raValue) || raValue < 0 || raValue >= 360) return "RA must be between 0 and 360 degrees.";
    if (!Number.isFinite(decValue) || decValue < -90 || decValue > 90) return "Dec must be between −90 and +90 degrees.";
    if (!Number.isFinite(radiusValue) || radiusValue <= 0) return "Radius must be greater than 0 arcsec.";
    if (!Number.isFinite(limitValue) || limitValue <= 0) return "Max sources per survey must be greater than 0.";
    if (selected.length === 0) return "Select at least one survey.";
    return null;
  }

  return (
    <Panel
      icon={Radar}
      title="Acquire"
      description="Cone search across the selected surveys. Raw files are extracted to Parquet and the downloads discarded."
    >
      <div className="grid gap-3 sm:grid-cols-4">
        <Field
          id="acquire-ra"
          label="RA (deg)"
          value={ra}
          onChange={setRa}
          min={0}
          max={359.999999}
          width="w-full"
          error={ra && (Number(ra) < 0 || Number(ra) >= 360 || !Number.isFinite(Number(ra))) ? "Use 0–360°" : undefined}
        />
        <Field
          id="acquire-dec"
          label="Dec (deg)"
          value={dec}
          onChange={setDec}
          min={-90}
          max={90}
          width="w-full"
          error={dec && (Number(dec) < -90 || Number(dec) > 90 || !Number.isFinite(Number(dec))) ? "Use −90–+90°" : undefined}
        />
        <Field
          id="acquire-radius"
          label="Radius (arcsec)"
          value={radius}
          onChange={setRadius}
          min={0.001}
          step="any"
          width="w-full"
          error={radius && (!(Number(radius) > 0) || !Number.isFinite(Number(radius))) ? "Must be greater than 0" : undefined}
        />
        <Field
          id="acquire-limit"
          label="Max sources / survey"
          value={limit}
          onChange={setLimit}
          min={1}
          step="1"
          width="w-full"
          error={limit && (!(Number(limit) > 0) || !Number.isFinite(Number(limit))) ? "Must be greater than 0" : undefined}
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {surveys.map((survey) => {
          const key = survey.name.toLowerCase();
          const on = selected.includes(key);
          return (
            <button
              key={key}
              type="button"
              onClick={() => toggle(key)}
              aria-pressed={on}
              className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] ${
                on
                  ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                  : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"
              }`}
            >
              {survey.name}
              <span className="ml-1.5 opacity-60">{survey.release}</span>
            </button>
          );
        })}
      </div>

      <Button
        onClick={run}
        disabled={running || !valid}
        icon={Play}
        tone="accent"
        className="mt-4 w-full"
      >
        {running ? "Acquisition running…" : "Run acquisition"}
      </Button>

      {projectId && Boolean(queryRegions?.length) && (
        <>
          <div className="my-3 border-t border-[var(--color-edge)]" />
          <Button
            onClick={runProject}
            disabled={running || selected.length === 0}
            icon={Radar}
            className="w-full"
          >
            {running ? "Acquisition running…" : `Acquire all project regions (${queryRegions?.length})`}
          </Button>
        </>
      )}

      {running && (
        <div className="mt-3 flex items-center justify-between gap-3" role="status" aria-live="polite">
          <Note>
            {progress?.message ?? "Archive queries can take several minutes. The window stays responsive."}
            {typeof progress?.items_done === "number" && typeof progress?.items_total === "number"
              ? ` (${progress.items_done}/${progress.items_total})`
              : ""}
          </Note>
          <Button onClick={cancel} icon={X} tone="default">
            Cancel
          </Button>
        </div>
      )}

      {formError && <div className="mt-3"><Note tone="warn">{formError}</Note></div>}
      {error && <div className="mt-3"><Note tone="bad">{error}</Note></div>}

      {result && ("regions" in result
        ? <ProjectResultTable result={result} />
        : <ResultTable result={result} />)}
    </Panel>
  );
}

function ProjectResultTable({ result }: { result: ProjectAcquisitionResult }) {
  return (
    <div className="mt-4 space-y-3">
      {result.regions.map((region, index) => (
        <div key={region.dataset_id}>
          <p className="px-1 pb-1 font-mono text-[11px] text-[var(--color-muted)]">
            region {index + 1}/{result.regions.length} · {region.query.ra_deg.toFixed(4)},{" "}
            {region.query.dec_deg.toFixed(4)}
          </p>
          <ResultTable result={region} />
        </div>
      ))}
      <Note>
        total across regions: {result.totals.curves} curves ·{" "}
        {result.totals.points.toLocaleString()} points · {result.totals.mb.toFixed(3)} MB
      </Note>
    </div>
  );
}

function ResultTable({ result }: { result: AcquisitionResult }) {
  return (
    <div className="mt-4">
      <Table head={["Survey", "Sources", "Curves", "Points", "MB"]}>
        {result.surveys.map((outcome, index) => (
          <tr
            key={outcome.survey}
            className={`border-b border-[var(--color-edge)]/50 ${index % 2 === 1 ? "bg-[var(--color-panel-2)]/40" : ""}`}
          >
            <td className="px-2.5 py-1.5">
              {outcome.survey}
              <span className="ml-1 opacity-50">{outcome.release}</span>
              {outcome.error && (
                <span className="ml-2">
                  <Badge tone="bad">failed</Badge>
                </span>
              )}
            </td>
            <td className="px-2.5 py-1.5 text-right">{outcome.sources_found}</td>
            <td className="px-2.5 py-1.5 text-right">{outcome.curves_stored}</td>
            <td className="px-2.5 py-1.5 text-right">
              {outcome.points_stored.toLocaleString()}
            </td>
            <td className="px-2.5 py-1.5 text-right">{outcome.mb_stored.toFixed(3)}</td>
          </tr>
        ))}
      </Table>

      <p className="mt-1.5 px-1 font-mono text-[11px] text-[var(--color-muted)]">
        dataset {result.dataset_id}
        <br />
        hash {result.content_hash?.slice(0, 24)}…
      </p>
    </div>
  );
}

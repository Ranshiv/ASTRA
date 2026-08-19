import { Play, Radar, X } from "lucide-react";
import { useEffect, useState } from "react";

import { SectionHeader } from "@/components/SectionHeader";
import { engine, type AcquisitionResult, type EngineJob, type SurveyInfo } from "@/lib/engine";

const ALL_SURVEYS = ["ztf", "gaia", "tess"];

export function AcquirePanel({ surveys, projectId }: { surveys: SurveyInfo[]; projectId?: string }) {
  const [ra, setRa] = useState("291.3663");
  const [dec, setDec] = useState("42.7844");
  const [radius, setRadius] = useState("10");
  const [selected, setSelected] = useState<string[]>(ALL_SURVEYS);
  const [job, setJob] = useState<EngineJob | null>(null);
  const [result, setResult] = useState<AcquisitionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = (name: string) =>
    setSelected((current) =>
      current.includes(name)
        ? current.filter((s) => s !== name)
        : [...current, name],
      );

  async function run() {
    setError(null);
    setResult(null);
    try {
      const params = {
        ra_deg: Number(ra),
        dec_deg: Number(dec),
        radius_arcsec: Number(radius),
        surveys: selected,
        limit: 5,
        project_id: projectId,
      };
      const key = ["acquire", params.ra_deg, params.dec_deg, params.radius_arcsec, ...[...selected].sort()].join(":");
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

  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.status)) return;

    let active = true;
    const poll = async () => {
      try {
        const next = await engine.jobStatus(job.job_id);
        if (!active) return;
        setJob(next);
        if (next.status === "completed") {
          setResult(next.result as AcquisitionResult);
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
    Number.isFinite(Number(ra)) &&
    Number.isFinite(Number(dec)) &&
    Number(radius) > 0 &&
    selected.length > 0;

  return (
    <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
      <SectionHeader
        icon={Radar}
        title="Acquire observations"
        description="Cone search across the selected surveys. Raw files are extracted to Parquet and the downloads discarded."
      />

      <div className="mt-4 grid grid-cols-3 gap-3">
        <Field label="RA (deg)" value={ra} onChange={setRa} />
        <Field label="Dec (deg)" value={dec} onChange={setDec} />
        <Field label="Radius (arcsec)" value={radius} onChange={setRadius} />
      </div>

      <div className="mt-3 flex gap-2">
        {surveys.map((survey) => {
          const key = survey.name.toLowerCase();
          const on = selected.includes(key);
          return (
            <button
              key={key}
              type="button"
              onClick={() => toggle(key)}
              className={`rounded-full border px-3 py-1 text-xs transition ${
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

      <button
        type="button"
        onClick={run}
        disabled={running || !valid}
        className="mt-4 flex w-full items-center justify-center gap-2 rounded border border-[var(--color-accent)]/60 px-3 py-2 text-sm
                   text-[var(--color-accent)] transition hover:bg-[var(--color-accent)]/10
                   disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Play size={14} strokeWidth={2} />
        {running ? "Acquisition running…" : "Run acquisition"}
      </button>

      {running && (
        <div className="mt-2 flex items-center justify-between gap-3 text-xs text-[var(--color-muted)]">
          <span>
            {progress?.message ?? "Archive queries can take several minutes. The window stays responsive."}
            {typeof progress?.items_done === "number" && typeof progress?.items_total === "number"
              ? ` (${progress.items_done}/${progress.items_total})`
              : ""}
          </span>
          <button
            type="button"
            onClick={cancel}
            className="flex items-center gap-1 rounded border border-[var(--color-edge)] px-2 py-1 text-[var(--color-warn)]"
          >
            <X size={12} strokeWidth={2} />
            Cancel
          </button>
        </div>
      )}

      {error && (
        <p className="mt-3 font-mono text-xs text-[var(--color-bad)]">{error}</p>
      )}

      {result && <ResultTable result={result} />}
    </section>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <label className="block">
      <span className="block text-xs text-[var(--color-muted)]">{label}</span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        inputMode="decimal"
        className="mt-1 w-full rounded border border-[var(--color-edge)] bg-[var(--color-void)]
                   px-2 py-1.5 text-sm outline-none focus:border-[var(--color-accent)]"
      />
    </label>
  );
}

function ResultTable({ result }: { result: AcquisitionResult }) {
  return (
    <div className="mt-4 overflow-hidden rounded border border-[var(--color-edge)]">
      <table className="w-full text-xs">
        <thead className="bg-[var(--color-panel-2)] text-[var(--color-muted)]">
          <tr>
            <th className="px-2 py-1.5 text-left font-normal">Survey</th>
            <th className="px-2 py-1.5 text-right font-normal">Sources</th>
            <th className="px-2 py-1.5 text-right font-normal">Curves</th>
            <th className="px-2 py-1.5 text-right font-normal">Points</th>
            <th className="px-2 py-1.5 text-right font-normal">MB</th>
          </tr>
        </thead>
        <tbody>
          {result.surveys.map((outcome, index) => (
            <tr
              key={outcome.survey}
              className={`border-t border-[var(--color-edge)]/50 ${index % 2 === 1 ? "bg-[var(--color-panel-2)]/40" : ""}`}
            >
              <td className="px-2 py-1.5">
                {outcome.survey}
                <span className="ml-1 opacity-50">{outcome.release}</span>
                {outcome.error && (
                  <span className="ml-2 rounded-full bg-[var(--color-bad)]/15 px-1.5 py-0.5 text-[var(--color-bad)]">
                    failed
                  </span>
                )}
              </td>
              <td className="px-2 py-1.5 text-right">{outcome.sources_found}</td>
              <td className="px-2 py-1.5 text-right">{outcome.curves_stored}</td>
              <td className="px-2 py-1.5 text-right">
                {outcome.points_stored.toLocaleString()}
              </td>
              <td className="px-2 py-1.5 text-right">{outcome.mb_stored.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="px-2 py-2 font-mono text-[11px] text-[var(--color-muted)]">
        dataset {result.dataset_id}
        <br />
        hash {result.content_hash?.slice(0, 24)}…
      </p>
    </div>
  );
}

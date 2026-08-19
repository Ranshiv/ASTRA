import { Activity, RefreshCcw, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

import { LightCurvePlot } from "@/components/LightCurvePlot";
import { SectionHeader } from "@/components/SectionHeader";
import {
  engine,
  type CurvePayload,
  type CurveSummary,
  type FoldedCurve,
} from "@/lib/engine";

export function CurveExplorer({ projectId }: { projectId?: string }) {
  const [curves, setCurves] = useState<CurveSummary[]>([]);
  const [selected, setSelected] = useState<CurvePayload | null>(null);
  const [folded, setFolded] = useState<FoldedCurve | null>(null);
  const [period, setPeriod] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    engine.curvesList(undefined, 500, projectId).then(setCurves).catch((err) => setError(String(err)));
  }, [projectId]);

  async function open(summary: CurveSummary) {
    setLoading(true);
    setFolded(null);
    setError(null);
    try {
      setSelected(await engine.curveGet(summary.path));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function applyFold() {
    if (!selected) return;
    const days = Number(period);
    if (!Number.isFinite(days) || days <= 0) return;
    try {
      setFolded(await engine.curveFold(selected.path, days));
    } catch (err) {
      setError(String(err));
    }
  }

  if (curves.length === 0) {
    return (
      <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
        <SectionHeader icon={Activity} title="Light curves" />
        <p className="mt-3 text-xs text-[var(--color-muted)]">
          No stored curves yet. Run an acquisition above.
        </p>
        {error && (
          <p className="mt-2 font-mono text-xs text-[var(--color-bad)]">{error}</p>
        )}
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
      <SectionHeader
        icon={Activity}
        title="Light curves"
        description={`${curves.length} stored`}
      />

      <div className="mt-3 max-h-96 overflow-y-auto rounded border border-[var(--color-edge)]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[var(--color-panel-2)] text-[var(--color-muted)]">
            <tr>
              <th className="px-2 py-1.5 text-left font-normal">Survey</th>
              <th className="px-2 py-1.5 text-left font-normal">Object</th>
              <th className="px-2 py-1.5 text-left font-normal">Band</th>
              <th className="px-2 py-1.5 text-right font-normal">Points</th>
              <th className="px-2 py-1.5 text-right font-normal">Span</th>
            </tr>
          </thead>
          <tbody>
            {curves.map((summary, index) => {
              const active = selected?.path === summary.path;
              return (
                <tr
                  key={summary.path}
                  onClick={() => open(summary)}
                  className={`cursor-pointer border-b border-[var(--color-edge)]/50 transition
                              hover:bg-[var(--color-accent)]/5 ${
                                active
                                  ? "bg-[var(--color-accent)]/10"
                                  : index % 2 === 1
                                    ? "bg-[var(--color-panel-2)]/40"
                                    : ""
                              }`}
                >
                  <td className="px-2 py-1.5">{summary.survey}</td>
                  <td className="px-2 py-1.5 font-mono">{summary.object_id}</td>
                  <td className="px-2 py-1.5">{summary.band}</td>
                  <td className="px-2 py-1.5 text-right">
                    {summary.points.toLocaleString()} pts
                  </td>
                  <td className="px-2 py-1.5 text-right text-[var(--color-muted)]">
                    {summary.time_span_days.toFixed(1)} d
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {loading && (
        <p className="mt-3 text-xs text-[var(--color-muted)]">Loading curve…</p>
      )}

      {selected && !loading && (
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-[var(--color-muted)]">
            <span>
              {selected.survey} {selected.object_id} · {selected.band} ·{" "}
              {selected.value_kind}
            </span>
            <span>
              {selected.downsampled
                ? `${selected.shown_points.toLocaleString()} of ${selected.points.toLocaleString()} shown`
                : `${selected.points.toLocaleString()} points`}
            </span>
          </div>

          <LightCurvePlot curve={selected} folded={folded} />

          <div className="flex items-center gap-2">
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
        </div>
      )}

      {error && (
        <p className="mt-2 font-mono text-xs text-[var(--color-bad)]">{error}</p>
      )}
    </section>
  );
}

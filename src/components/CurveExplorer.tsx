import { Activity, RefreshCcw, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

import { LightCurvePlot } from "@/components/LightCurvePlot";
import {
  engine,
  type CurvePayload,
  type CurveSummary,
  type FoldedCurve,
} from "@/lib/engine";
import { Button, Empty, Field, Note, Panel, Table } from "@/components/ui";

export function CurveExplorer({ projectId }: { projectId?: string }) {
  const [curves, setCurves] = useState<CurveSummary[]>([]);
  const [selected, setSelected] = useState<CurvePayload | null>(null);
  const [folded, setFolded] = useState<FoldedCurve | null>(null);
  const [period, setPeriod] = useState("");
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    engine.curvesList(undefined, 500, projectId).then(setCurves).catch((err) => setError(String(err)));
  }, [projectId]);

  const visibleCurves = curves.filter((curve) =>
    `${curve.survey} ${curve.object_id} ${curve.band}`.toLowerCase().includes(query.trim().toLowerCase()),
  );

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
      <Panel icon={Activity} title="Light curves">
        <Empty>No stored curves yet. Acquire a sky region above.</Empty>
        {error && <Note tone="bad">{error}</Note>}
      </Panel>
    );
  }

  return (
    <Panel icon={Activity} title="Light curves" description={`${curves.length} stored`}>
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <Field
          label="Filter curves"
          value={query}
          onChange={setQuery}
          placeholder="Search survey, object, or band…"
          width="w-72"
          id="curve-filter"
        />
        <Note>Showing {visibleCurves.length} of {curves.length}</Note>
      </div>

      <div className="max-h-96 overflow-y-auto">
        <Table head={["Survey", "Object", "Band", "Points", "Span"]}>
          {visibleCurves.map((summary, index) => {
            const active = selected?.path === summary.path;
            return (
              <tr
                key={summary.path}
                onClick={() => open(summary)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    void open(summary);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-label={`Open ${summary.survey} ${summary.object_id} ${summary.band} light curve`}
                aria-selected={active}
                className={`cursor-pointer border-b border-[var(--color-edge)]/50 transition
                            hover:bg-[var(--color-accent)]/5 ${
                              active
                                ? "bg-[var(--color-accent)]/10"
                                : index % 2 === 1
                                  ? "bg-[var(--color-panel-2)]/40"
                                  : ""
                            }`}
              >
                <td className="px-2.5 py-1.5">{summary.survey}</td>
                <td className="px-2.5 py-1.5 font-mono text-xs">{summary.object_id}</td>
                <td className="px-2.5 py-1.5">{summary.band}</td>
                <td className="px-2.5 py-1.5 text-right">
                  {summary.points.toLocaleString()} pts
                </td>
                <td className="px-2.5 py-1.5 text-right text-[var(--color-muted)]">
                  {summary.time_span_days.toFixed(1)} d
                </td>
              </tr>
            );
          })}
        </Table>
      </div>

      {loading && <Note>Loading curve…</Note>}

      {selected && !loading && (
        <div className="mt-3 flex flex-col gap-2">
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

          <div className="flex items-end gap-2">
            <Field
              label="Fold period (days)"
              value={period}
              onChange={setPeriod}
              placeholder="e.g. 3.14"
              width="w-40"
              id="fold-period"
              type="text"
            />
            <Button
              onClick={() => void applyFold()}
              disabled={!Number.isFinite(Number(period)) || Number(period) <= 0}
              icon={RefreshCcw}
            >
              Fold
            </Button>
            {folded && (
              <Button onClick={() => setFolded(null)} icon={RotateCcw}>
                Unfold
              </Button>
            )}
          </div>
        </div>
      )}

      {error && <Note tone="bad">{error}</Note>}
    </Panel>
  );
}

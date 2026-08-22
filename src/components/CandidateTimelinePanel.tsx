import { useEffect, useState } from "react";

import { LightCurvePlot } from "@/components/LightCurvePlot";
import { engine, type CandidateTimeline, type CandidateTimelineCurve, type CurvePayload } from "@/lib/engine";

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

export function CandidateTimelinePanel({
  candidateId,
  projectId,
}: {
  candidateId: string;
  projectId?: string;
}) {
  const [timeline, setTimeline] = useState<CandidateTimeline | null>(null);

  useEffect(() => {
    let cancelled = false;
    setTimeline(null);
    engine
      .candidateTimeline(candidateId, "default", projectId)
      .then((result) => {
        if (!cancelled) setTimeline(result);
      })
      .catch(() => {
        /* timeline is an enhancement; preserve the curve view */
      });
    return () => {
      cancelled = true;
    };
  }, [candidateId, projectId]);

  return (
    <details className="group rounded border border-[var(--color-edge)] p-2" open={Boolean(timeline)}>
      <summary className="cursor-pointer list-none text-xs font-medium [&::-webkit-details-marker]:hidden">
        Combined observation timeline
      </summary>
      {timeline?.warning && <p className="mt-1 text-xs text-[var(--color-warn)]">{timeline.warning}</p>}
      {timeline && timeline.events.length > 0 ? (
        <div className="mt-2 space-y-2">
          {timeline.events.map((event) => {
            const span = Math.max(event.time_end - event.time_start, 1e-6);
            return (
              <div key={`${event.survey}-${event.release}-${event.object_id}-${event.band}`} className="text-xs">
                <div className="flex justify-between gap-2 text-[var(--color-muted)]">
                  <span>{event.survey} · {event.release} · {event.band}</span>
                  <span>{event.points.toLocaleString()} pts</span>
                </div>
                <div
                  className="mt-1 h-2 overflow-hidden rounded bg-[var(--color-panel-2)]"
                  title={`${event.time_start.toFixed(3)}–${event.time_end.toFixed(3)} BJD_TDB`}
                >
                  <div
                    className={`h-full rounded ${event.resolved ? "bg-[var(--color-accent)]" : "bg-[var(--color-warn)]"}`}
                    style={{
                      width: `${Math.max(4, Math.min(100, (span / Math.max(...timeline.events.map((item) => item.time_end - item.time_start), 1e-6)) * 100))}%`,
                    }}
                  />
                </div>
              </div>
            );
          })}
          {timeline.curves.length > 1 && (
            <div className="mt-3 grid gap-2 lg:grid-cols-2">
              {timeline.curves.slice(0, 6).map((item) => (
                <div
                  key={`plot-${item.survey}-${item.release}-${item.object_id}-${item.band}`}
                  className="rounded border border-[var(--color-edge)]/60 p-1"
                >
                  <p className="px-1 py-1 text-[11px] text-[var(--color-muted)]">
                    {item.survey} · {item.band} · {item.resolved ? "resolved" : "blended"}
                  </p>
                  <LightCurvePlot curve={timelineCurvePayload(item)} folded={null} />
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <p className="mt-1 text-xs text-[var(--color-muted)]">No matching multi-survey curves were found.</p>
      )}
    </details>
  );
}

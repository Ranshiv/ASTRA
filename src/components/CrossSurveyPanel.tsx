/** Plan section 15's cross-survey evidence, which had no interface at all.
 *
 * Two distinctions are load-bearing here and are shown rather than smoothed
 * over. `resolved_surveys` is not `independent_surveys`: TESS pixels span
 * about 21 arcsec, so a TESS match corroborates the neighbourhood, not the
 * object, and counting it would manufacture agreement for every star near a
 * bright variable. And `weight_used` says how much of the evidence a
 * consistency score was actually computed from — 0.70 from 90% of the weight
 * is not the same number as 0.70 from all of it.
 */
import { Link2, Radar } from "lucide-react";
import { useMemo, useState } from "react";

import { engine, type CrossSurveyProfile } from "@/lib/engine";
import { Badge, Button, Empty, KeyValue, Note, Panel, Select, StatTile, Table, num, useAsync } from "@/components/ui";

export function SurveyViews({ profile }: { profile: CrossSurveyProfile }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={profile.resolved_surveys > 1 ? "ok" : "warn"}>
          {profile.resolved_surveys} resolved
        </Badge>
        <Badge tone="muted">{profile.independent_surveys} detected</Badge>
        <Badge tone="accent">consistency {num(profile.consistency, 3)}</Badge>
        <Badge tone={profile.weight_used >= 0.999 ? "muted" : "warn"}>
          weight used {num(profile.weight_used, 2)}
        </Badge>
        {profile.period_fap !== null && (
          <Badge tone={profile.period_fap > 0.1 ? "warn" : "muted"}>
            {`period FAP ${(profile.period_fap * 100).toFixed(1)}%`}
          </Badge>
        )}
        {profile.blended.map((survey) => (
          <Badge key={survey} tone="bad">
            {survey} blended
          </Badge>
        ))}
        {profile.ambiguous.map((survey) => (
          <Badge key={survey} tone="warn">
            {survey} ambiguous
          </Badge>
        ))}
      </div>

      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {profile.views.map((view) => (
          <div
            key={`${view.survey}-${view.object_id}-${view.band}`}
            className="rounded border border-[var(--color-edge)] bg-[var(--color-panel-2)]/40 p-2"
          >
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium">
                {view.survey} · {view.band}
              </span>
              {profile.blended.includes(view.survey) && <Badge tone="bad">blended</Badge>}
            </div>
            <p className="mb-1 truncate font-mono text-[10px] text-[var(--color-muted)]">
              {view.object_id}
            </p>
            <KeyValue
              rows={[
                ["Points", view.points.toLocaleString()],
                ["Period (d)", num(view.best_period_days, 6)],
                ["Period S/N", num(view.period_snr, 2)],
                ["Reduced χ²", num(view.reduced_chi2, 2)],
                ["Amplitude", `${num(view.robust_amplitude, 4)} ${view.value_kind}`],
                ["Fractional", num(view.fractional_amplitude, 4)],
                ["Baseline (d)", num(view.baseline_days, 1)],
              ]}
            />
          </div>
        ))}
      </div>

      {Object.keys(profile.components).length > 0 && (
        <Table head={["Evidence component", "Score"]}>
          {Object.entries(profile.components).map(([name, value]) => (
            <tr key={name} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{name.replace(/_/g, " ")}</td>
              <td className="px-2 py-1.5">{num(value, 4)}</td>
            </tr>
          ))}
        </Table>
      )}

      {profile.notes.length > 0 && (
        <ul className="list-inside list-disc text-xs text-[var(--color-muted)]">
          {profile.notes.map((note, index) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function CrossSurveyPanel({ projectId }: { projectId?: string }) {
  const [anchorSurvey, setAnchorSurvey] = useState("");
  const groups = useAsync(
    () => engine.crossmatch(undefined, projectId, anchorSurvey || undefined),
    [projectId, anchorSurvey],
  );
  const profiles = useAsync(
    () => engine.profiles(undefined, undefined, projectId, anchorSurvey || undefined),
    [projectId, anchorSurvey],
  );
  const anchorOptions = useMemo(() => {
    const counts = groups.data?.summary.grouping_bias?.survey_counts ?? {};
    return [
      { value: "", label: "Automatic · largest catalogue" },
      ...Object.keys(counts).sort().map((survey) => ({ value: survey, label: `Explicit · ${survey}` })),
    ];
  }, [groups.data]);

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Link2}
        title="Cross-survey matching"
        description="Positional grouping of every stored source, proper-motion aware."
        actions={<Button onClick={() => { void groups.reload(); void profiles.reload(); }}>Refresh</Button>}
      >
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <Select
            label="Grouping anchor"
            value={anchorSurvey}
            options={anchorOptions}
            onChange={setAnchorSurvey}
          />
          <Note>
            {anchorSurvey
              ? `Reproducible population anchored on ${anchorSurvey}.`
              : "Default uses the largest catalogue; ties are resolved lexically."}
          </Note>
        </div>
        {groups.error && <Note tone="bad">{groups.error}</Note>}
        {groups.data && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatTile label="Groups" value={groups.data.summary.groups.toLocaleString()} />
              <StatTile label="Multi-survey" value={groups.data.summary.multi_survey.toLocaleString()} />
              <StatTile
                label="Resolved multi-survey"
                value={groups.data.summary.resolved_multi_survey.toLocaleString()}
                tone="accent"
              />
              <StatTile
                label="Ambiguous"
                value={groups.data.summary.ambiguous.toLocaleString()}
                tone={groups.data.summary.ambiguous > 0 ? "warn" : "neutral"}
              />
            </div>
            <div className="mt-3 flex flex-col gap-2">
              <Note>
                Resolved multi-survey is the honest count. A blended match corroborates the
                neighbourhood at that survey's pixel scale, not this object.
              </Note>
              {groups.data.summary.grouping_bias && (
                <div className="rounded-lg border border-[var(--color-warn)]/40 bg-[var(--color-warn)]/5 px-3 py-2">
                  <Note tone="warn">
                    {groups.data.summary.grouping_bias.warning} · anchor share{" "}
                    {num(groups.data.summary.grouping_bias.anchor_share, 2)}.
                  </Note>
                </div>
              )}
            </div>
            {groups.data.groups.length > 0 && (
              <div className="mt-3">
                <Table head={["Surveys", "Resolved", "Members", "Separations", "Flags"]}>
                {groups.data.groups.map((group, index) => (
                  <tr key={index} className="border-b border-[var(--color-edge)]/50">
                    <td className="px-2 py-1.5">{group.surveys.join(" + ")}</td>
                    <td className="px-2 py-1.5 tabular-nums">{group.resolved_surveys}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px] text-[var(--color-muted)]">
                      {Object.entries(group.members)
                        .map(([survey, id]) => `${survey}:${id}`)
                        .join("  ")}
                    </td>
                    <td className="px-2 py-1.5 text-[var(--color-muted)]">
                      {Object.entries(group.separations_arcsec)
                        .map(([survey, sep]) => `${survey} ${sep.toFixed(2)}″`)
                        .join(", ")}
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="flex gap-1">
                        {group.blended.map((survey) => (
                          <Badge key={survey} tone="bad">
                            {survey}
                          </Badge>
                        ))}
                        {group.ambiguous.map((survey) => (
                          <Badge key={survey} tone="warn">
                            {survey}
                          </Badge>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
                </Table>
              </div>
            )}
          </>
        )}
      </Panel>

      <Panel
        icon={Radar}
        title="Evidence profiles"
        description={
          profiles.loading
            ? "Assembling cross-survey evidence…"
            : `${profiles.data?.profiled ?? 0} profiled, ranked by consistency`
        }
        actions={<Button onClick={() => void profiles.reload()}>Refresh</Button>}
      >
        {profiles.error && <Note tone="bad">{profiles.error}</Note>}
        {!profiles.loading && (profiles.data?.profiles.length ?? 0) === 0 && (
          <Empty>
            No multi-survey groups to profile. Acquire more than one survey over the same
            positions.
          </Empty>
        )}
        <div className="flex flex-col gap-4">
          {profiles.data?.profiles.map((profile, index) => (
            <div key={index} className="rounded border border-[var(--color-edge)] p-2">
              <SurveyViews profile={profile} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

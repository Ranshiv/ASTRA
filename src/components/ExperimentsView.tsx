/** Plan section 19: every experiment reproducible, and section 20's groups.
 *
 * The engine has recorded provenance, drift verification and cross-experiment
 * comparison since Phase 8; none of it was reachable from the interface. The
 * two things this view refuses to smooth over are the two that would mislead a
 * researcher: an experiment whose environment has moved is shown as drifted
 * field by field, and a comparison spanning different feature or preprocessing
 * versions is labelled not comparable rather than quietly ranked.
 */
import {
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  GitCompare,
  Layers,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { useState } from "react";

import {
  engine,
  type ExperimentComparison,
  type ExperimentRecord,
  type ExperimentVerification,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Select, Table, num, useAction, useAsync } from "@/components/ui";

const METRICS = ["roc_auc", "average_precision", "precision", "recall", "f1"];

export function ExperimentsView({ projectId }: { projectId?: string }) {
  const { data: experiments, error, loading, reload } = useAsync(
    () => engine.experiments(projectId), [projectId],
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [record, setRecord] = useState<ExperimentRecord | null>(null);
  const [verification, setVerification] = useState<ExperimentVerification | null>(null);
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [metric, setMetric] = useState("roc_auc");
  const [survey, setSurvey] = useState("ztf");
  const [fraction, setFraction] = useState("0.1");
  const run = useAction("Ablations are long-running research actions, not a refresh.");

  function toggle(id: string) {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  async function open(id: string) {
    setVerification(null);
    setRecord(await engine.experiment(id));
  }

  async function verify(id: string) {
    setVerification(await engine.experimentVerify(id));
  }

  async function compare() {
    setComparison(await engine.experimentCompare(selected, metric));
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={FlaskConical}
        title="Experiments"
        description={
          loading
            ? "Reading experiment records…"
            : `${experiments?.length ?? 0} recorded runs`
        }
        actions={
          <>
            <Button icon={Layers} onClick={() => void reload()}>
              Refresh
            </Button>
            <Button
              icon={Timer}
              disabled={run.busy}
              onClick={() =>
                void run.run("Running the ablation suite; minutes, not seconds…", async () => {
                  const result = await engine.ablation(Number(fraction) || 0.1, 42, survey || undefined, projectId);
                  await reload();
                  return `Recorded ${result.experiment_id}.`;
                })
              }
            >
              Run ablation
            </Button>
            <Button
              icon={Timer}
              disabled={run.busy}
              onClick={() =>
                void run.run("Running five independent injection seeds…", async () => {
                  const result = await engine.ablationRepeated(
                    Number(fraction) || 0.1,
                    [17, 29, 43, 59, 71],
                    survey || undefined,
                    projectId,
                  );
                  await reload();
                  return `Recorded ${result.experiment_id} with seed intervals.`;
                })
              }
            >
              Repeated seeds
            </Button>
          </>
        }
      >
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <Field label="Injected fraction" value={fraction} onChange={setFraction} width="w-20" />
          <Field
            label="Stratify by survey"
            value={survey}
            onChange={setSurvey}
            placeholder="blank = pooled"
            width="w-32"
          />
          <Note>
            Leaving the survey blank pools ZTF and TESS. Those populations are structurally
            different, so the detectors partly separate by survey rather than by behaviour — the
            same comparison scores about 0.79 on ZTF alone and about 0.63 pooled.
          </Note>
        </div>

        {error && <Note tone="bad">{error}</Note>}
        {run.status && <Note>{run.status}</Note>}

        {!loading && !error && (experiments?.length ?? 0) === 0 && (
          <Empty>No experiments recorded yet. Run an ablation, or generate candidates.</Empty>
        )}

        {(experiments?.length ?? 0) > 0 && (
          <Table head={["", "Experiment", "Kind", "Recorded", "Code", "Runtime", ""]}>
            {experiments!.map((item) => (
              <tr key={item.experiment_id} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5">
                  <input
                    type="checkbox"
                    aria-label={`Compare ${item.experiment_id}`}
                    checked={selected.includes(item.experiment_id)}
                    onChange={() => toggle(item.experiment_id)}
                  />
                </td>
                <td className="px-2 py-1.5 font-mono">{item.experiment_id}</td>
                <td className="px-2 py-1.5">{item.kind}</td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">{item.created_utc}</td>
                <td className="px-2 py-1.5 font-mono text-[var(--color-muted)]">
                  {item.code_version?.slice(0, 8)}
                </td>
                <td className="px-2 py-1.5">{num(item.runtime_seconds, 1)}s</td>
                <td className="px-2 py-1.5">
                  <div className="flex gap-1.5">
                    {item.failed && <Badge tone="bad">failed</Badge>}
                    <Button onClick={() => void open(item.experiment_id)}>Open</Button>
                    <Button icon={ShieldCheck} onClick={() => void verify(item.experiment_id)}>
                      Verify
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        )}
      </Panel>

      {verification && (
        <Panel
          icon={verification.reproducible ? CheckCircle2 : AlertTriangle}
          title={`Reproducibility · ${verification.experiment_id}`}
          description={verification.note}
        >
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <Badge tone={verification.reproducible ? "ok" : "warn"}>
                {verification.reproducible ? "reproducible" : "environment drifted"}
              </Badge>
              <span className="text-xs text-[var(--color-muted)]">seed {verification.seed}</span>
            </div>
            {Object.keys(verification.drift).length > 0 && (
              <Table head={["Field", "Recorded", "Current"]}>
                {Object.entries(verification.drift).flatMap(([field, value]) => {
                  const entry = value as Record<string, unknown>;
                  // `environment` nests one level deeper: library -> recorded/current.
                  if (entry.recorded === undefined && entry.current === undefined) {
                    return Object.entries(entry).map(([library, versions]) => {
                      const pair = versions as { recorded?: unknown; current?: unknown };
                      return (
                        <tr key={`${field}.${library}`} className="border-b border-[var(--color-edge)]/50">
                          <td className="px-2 py-1.5 font-mono">{`${field}.${library}`}</td>
                          <td className="px-2 py-1.5 font-mono">{String(pair.recorded ?? "—")}</td>
                          <td className="px-2 py-1.5 font-mono text-[var(--color-warn)]">
                            {String(pair.current ?? "—")}
                          </td>
                        </tr>
                      );
                    });
                  }
                  return [
                    <tr key={field} className="border-b border-[var(--color-edge)]/50">
                      <td className="px-2 py-1.5 font-mono">{field}</td>
                      <td className="px-2 py-1.5 font-mono">{String(entry.recorded ?? "—")}</td>
                      <td className="px-2 py-1.5 font-mono text-[var(--color-warn)]">
                        {String(entry.current ?? "—")}
                      </td>
                    </tr>,
                  ];
                })}
              </Table>
            )}
          </div>
        </Panel>
      )}

      <Panel
        icon={GitCompare}
        title="Compare"
        description={`${selected.length} selected`}
        actions={
          <>
            <Select
              label="Metric"
              value={metric}
              onChange={setMetric}
              options={METRICS.map((name) => ({ value: name, label: name }))}
            />
            <Button icon={GitCompare} disabled={selected.length < 2} onClick={() => void compare()}>
              Compare
            </Button>
          </>
        }
      >
        {!comparison && <Empty>Select two or more experiments above.</Empty>}
        {comparison && (
          <div className="flex flex-col gap-2">
            {!comparison.comparable && (
              <div className="rounded border border-[var(--color-warn)]/40 bg-[var(--color-warn)]/5 p-2">
                <Note tone="warn">
                  {comparison.warning ??
                    "These experiments span different feature or preprocessing versions."}{" "}
                  A metric computed from different inputs is not the same metric, so the ranking
                  below is not a like-for-like result.
                </Note>
              </div>
            )}
            <Table head={["Experiment", "Kind", comparison.metric, "Feature v", "Runtime"]}>
              {comparison.rows.map((row) => (
                <tr
                  key={row.experiment_id}
                  className={`border-b border-[var(--color-edge)]/50 ${
                    comparison.best?.experiment_id === row.experiment_id && comparison.comparable
                      ? "text-[var(--color-accent)]"
                      : ""
                  }`}
                >
                  <td className="px-2 py-1.5 font-mono">{row.experiment_id}</td>
                  <td className="px-2 py-1.5">{row.kind}</td>
                  <td className="px-2 py-1.5">{num(row.value, 4)}</td>
                  <td className="px-2 py-1.5">{row.feature_version}</td>
                  <td className="px-2 py-1.5">{num(row.runtime_seconds, 1)}s</td>
                </tr>
              ))}
            </Table>
          </div>
        )}
      </Panel>

      {record && (
        <Panel
          icon={FlaskConical}
          title={record.provenance.experiment_id}
          description={record.notes || record.kind}
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">Provenance</h3>
              <KeyValue
                rows={[
                  ["Recorded", record.provenance.created_utc],
                  ["Code version", record.provenance.code_version],
                  ["Feature version", record.provenance.feature_version],
                  ["Feature schema", record.provenance.feature_schema_hash?.slice(0, 16)],
                  ["Preprocessing", record.provenance.preprocessing_version],
                  ["Dataset", record.provenance.dataset_id ?? "—"],
                  ["Seed", record.provenance.seed],
                  ["Runtime", `${num(record.runtime_seconds, 2)}s`],
                ]}
              />
            </div>
            <div>
              <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">Configuration</h3>
              <pre className="max-h-64 overflow-auto rounded border border-[var(--color-edge)] bg-[var(--color-void)] p-2 text-[11px]">
                {JSON.stringify(record.configuration, null, 2)}
              </pre>
            </div>
          </div>
          <h3 className="mb-1.5 mt-4 text-xs font-medium text-[var(--color-muted)]">Results</h3>
          <pre className="max-h-96 overflow-auto rounded border border-[var(--color-edge)] bg-[var(--color-void)] p-2 text-[11px]">
            {JSON.stringify(record.results, null, 2)}
          </pre>
        </Panel>
      )}
    </div>
  );
}

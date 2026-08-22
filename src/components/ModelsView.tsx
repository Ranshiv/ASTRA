/** Models: feature matrices, baseline detection, rankers and deep models.
 *
 * A released ASTRA installer ships a CPU-only engine that deliberately excludes
 * PyTorch, so `deepTrain` and `deepCompare(includeDeep)` genuinely cannot run
 * there. That refusal is rendered as the engine's own explanation rather than a
 * raw error, because "No module named 'torch'" reads like a broken install when
 * it is a deliberate 3.5 GB trade-off.
 */
import { Boxes, Brain, Cpu, Eraser, Gauge, Layers, SlidersHorizontal, Sparkles } from "lucide-react";
import { useState } from "react";

import {
  engine,
  type DeepComparison,
  type DeepTrainReport,
  type DetectionResult,
  type RankerResult,
  type SweepResult,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Select, Table, num, useAction, useAsync } from "@/components/ui";

/** The engine's refusal is long and explanatory; treat it as prose, not an error. */
function isDeepUnavailable(message: string): boolean {
  return message.includes("PyTorch is not available in this build");
}

export function ModelsView({ projectId }: { projectId?: string }) {
  const matrices = useAsync(() => engine.featuresList(projectId), [projectId]);
  // Ranker models and library versions are engine-wide, not per-project.
  const rankers = useAsync(() => engine.rankerList());
  const versions = useAsync(() => engine.versions());

  const [survey, setSurvey] = useState("ztf");
  const [matrixName, setMatrixName] = useState("default");
  const [kind, setKind] = useState<"autoencoder" | "vae" | "transformer">("autoencoder");
  const [epochs, setEpochs] = useState("30");
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [training, setTraining] = useState<DeepTrainReport | null>(null);
  const [comparison, setComparison] = useState<DeepComparison | null>(null);
  const [ranker, setRanker] = useState<RankerResult | null>(null);
  const [sweep, setSweep] = useState<SweepResult | null>(null);
  const [deepRefusal, setDeepRefusal] = useState<string | null>(null);

  const features = useAction();
  const deep = useAction();
  const supervised = useAction();

  async function runDeep(message: string, work: () => Promise<string>) {
    setDeepRefusal(null);
    await deep.run(message, async () => {
      try {
        return await work();
      } catch (err) {
        const message = String(err);
        if (isDeepUnavailable(message)) {
          setDeepRefusal(message);
          return "This build cannot run deep models — see below.";
        }
        throw err;
      }
    });
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Layers}
        title="Feature matrices"
        description="Extraction runs a Lomb-Scargle period search per curve, which profiling measured at 98.6% of pipeline time. Cached rows are reused."
        actions={
          <>
            <Button
              icon={Layers}
              disabled={features.busy}
              tone="accent"
              onClick={() =>
                void features.run("Extracting features across the store…", async () => {
                  const result = await engine.featuresBuild(matrixName, survey || undefined);
                  await matrices.reload();
                  return `${result.rows} rows, ${result.usable_rows} usable, feature version ${result.feature_version}.`;
                })
              }
            >
              Build matrix
            </Button>
            <Button
              icon={Layers}
              disabled={features.busy}
              onClick={() =>
                void features.run("Extracting checkpointed batches across the store…", async () => {
                  const result = await engine.featuresBuildResumable(matrixName, survey || undefined);
                  await matrices.reload();
                  return `${result.completed}/${result.source_count} curves completed in ${result.batches} durable batches${result.resumed ? " (resumed)" : ""}.`;
                })
              }
            >
              Build resumable
            </Button>
            <Button
              icon={Gauge}
              disabled={features.busy}
              onClick={() =>
                void features.run("Running the baseline ensemble…", async () => {
                  const result = await engine.detect(matrixName);
                  setDetection(result);
                  return `${result.rows_scored} rows scored, ${result.rows_skipped} skipped.`;
                })
              }
            >
              Run detection
            </Button>
            <Button
              icon={Eraser}
              onClick={() =>
                void features.run("Clearing the feature cache…", async () => {
                  const { cleared } = await engine.featureCacheClear();
                  return `${cleared} cached feature rows removed.`;
                })
              }
            >
              Clear cache
            </Button>
          </>
        }
      >
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <Field label="Matrix name" value={matrixName} onChange={setMatrixName} />
          <Field label="Survey" value={survey} onChange={setSurvey} placeholder="blank = all" />
        </div>
        {features.status && <Note>{features.status}</Note>}
        {matrices.error && <Note tone="bad">{matrices.error}</Note>}
        {(matrices.data?.length ?? 0) === 0 ? (
          <Empty>No feature matrices built yet.</Empty>
        ) : (
          <Table head={["Name", "Rows", "Size", "Path"]}>
            {matrices.data!.map((item) => (
              <tr key={item.path} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5 font-mono">{item.name}</td>
                <td className="px-2 py-1.5">{item.rows.toLocaleString()}</td>
                <td className="px-2 py-1.5">{num(item.mb, 2)} MB</td>
                <td className="px-2 py-1.5 truncate font-mono text-[var(--color-muted)]">{item.path}</td>
              </tr>
            ))}
          </Table>
        )}

        {detection && (
          <div className="mt-3">
            <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">
              Detectors · consensus is a mean of RANKS, not of scores
            </h3>
            <Note>
              Rank aggregation is invariant to a detector's score distribution, so a plateaued
              detector cannot distort the consensus just by having an unusual shape. Scores are
              min-max normalised per run: 1.0 means most anomalous in this batch, not an absolute
              level.
            </Note>
            <Table head={["Detector", "Flagged", "Mean score", "Max score"]}>
              {detection.detectors.map((item) => (
                <tr key={item.name} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{item.name}</td>
                  <td className="px-2 py-1.5">{item.flagged}</td>
                  <td className="px-2 py-1.5">{num(item.score_mean, 4)}</td>
                  <td className="px-2 py-1.5">{num(item.score_max, 4)}</td>
                </tr>
              ))}
            </Table>
          </div>
        )}
      </Panel>

      <Panel
        icon={Sparkles}
        title="Supervised ranker"
        description="Trained from interesting/needs_follow_up against artifact/known_object labels, behind a deliberate data gate."
        actions={
          <>
            <Button
              icon={Sparkles}
              disabled={supervised.busy}
              tone="accent"
              onClick={() =>
                void supervised.run("Checking the label gate and fitting…", async () => {
                  const result = await engine.rankerTrain();
                  setRanker(result);
                  await rankers.reload();
                  return result.ready
                    ? `Trained ${result.model_name}.`
                    : (result.reason ?? "Gate not met.");
                })
              }
            >
              Train
            </Button>
            <Button
              disabled={supervised.busy}
              onClick={() =>
                void supervised.run("Applying the trained ranker…", async () => {
                  const result = await engine.rankerApply();
                  return result.ready ? "Applied and re-ranked." : (result.reason ?? "Not applied.");
                })
              }
            >
              Apply
            </Button>
          </>
        }
      >
        {supervised.status && <Note>{supervised.status}</Note>}
        {ranker && !ranker.ready && (
          <div className="mt-2 rounded border border-[var(--color-warn)]/40 bg-[var(--color-warn)]/5 p-2">
            <Note tone="warn">
              {ranker.reason ?? "Not enough labels yet."} A new installation is intentionally
              untrainable until it has 50 usable labels, 10 per class, across two independent object
              groups per class.
            </Note>
            {ranker.gate && (
              <pre className="mt-1 overflow-auto text-[11px]">{JSON.stringify(ranker.gate, null, 2)}</pre>
            )}
          </div>
        )}
        {rankers.error && <Note tone="bad">{rankers.error}</Note>}
        {(rankers.data?.length ?? 0) === 0 ? (
          <Empty>No ranker models saved.</Empty>
        ) : (
          <Table head={["Model", "Kind", "Trained", "Checksum"]}>
            {rankers.data!.map((item) => {
              const row = item as unknown as Record<string, string>;
              return (
                <tr key={row.model_name} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{row.model_name}</td>
                  <td className="px-2 py-1.5">{row.kind}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">{row.created_utc}</td>
                  <td className="px-2 py-1.5 font-mono text-[var(--color-muted)]">
                    {row.model_sha256?.slice(0, 12)}
                  </td>
                </tr>
              );
            })}
          </Table>
        )}
      </Panel>

      <Panel
        icon={Brain}
        title="Deep models"
        description="Patch transformer, convolutional autoencoder and VAE over resampled sequences."
        actions={
          <>
            <Select
              label="Kind"
              value={kind}
              onChange={(value) => setKind(value as "autoencoder" | "vae" | "transformer")}
              options={[
                { value: "autoencoder", label: "autoencoder" },
                { value: "vae", label: "VAE" },
                { value: "transformer", label: "patch transformer" },
              ]}
            />
            <Field label="Epochs" value={epochs} onChange={setEpochs} width="w-16" />
            <Button
              icon={Brain}
              disabled={deep.busy}
              tone="accent"
              onClick={() =>
                void runDeep("Training; minutes, not seconds…", async () => {
                  const report = await engine.deepTrain(
                    "default", kind, survey || undefined, Number(epochs) || 30,
                  );
                  setTraining(report);
                  return report.error
                    ? report.error
                    : `${report.kind} trained on ${report.device}: ${report.parameters.toLocaleString()} parameters, best epoch ${report.best_epoch}.`;
                })
              }
            >
              Train deep model
            </Button>
            <Button
              icon={Cpu}
              disabled={deep.busy}
              onClick={() =>
                void runDeep("Running injection recovery across methods…", async () => {
                  const result = await engine.deepCompare(survey || undefined, 0.1, Number(epochs) || 20);
                  setComparison(result);
                  return result.error ?? `Best method: ${result.best_method ?? "none"}.`;
                })
              }
            >
              Compare methods
            </Button>
            <Button
              icon={SlidersHorizontal}
              disabled={deep.busy}
              onClick={() =>
                void runDeep("Sweeping hyperparameters across seeds; this is the longest action here…", async () => {
                  const result = await engine.deepSweep(
                    kind, survey || undefined, [17, 29, 43], Number(epochs) || 20,
                  );
                  setSweep(result);
                  return result.separated
                    ? `Best configuration separated from the runner-up (${result.experiment_id}).`
                    : result.note || "No configuration separated across these seeds.";
                })
              }
            >
              Sweep
            </Button>
          </>
        }
      >
        {deep.status && <Note>{deep.status}</Note>}

        {deepRefusal && (
          <div className="mt-2 rounded border border-[var(--color-accent)]/40 bg-[var(--color-accent)]/5 p-3">
            <div className="mb-1 flex items-center gap-2">
              <Badge tone="accent">CPU-only build</Badge>
            </div>
            <p className="text-xs leading-relaxed text-[var(--color-text)]">{deepRefusal}</p>
          </div>
        )}

        {training && !training.error && (
          <div className="mt-3">
            <KeyValue
              rows={[
                ["Kind", training.kind],
                ["Device", `${training.device} — ${training.device_reason}`],
                ["Parameters", training.parameters.toLocaleString()],
                ["Epochs run", `${training.epochs_run} (best ${training.best_epoch})`],
                ["Best val loss", num(training.best_val_loss, 6)],
                ["Batch size", training.batch_size],
                ["Seconds", num(training.seconds, 1)],
                ["Checkpoint", training.checkpoint ?? "—"],
              ]}
            />
          </div>
        )}

        {comparison && !comparison.error && (
          <div className="mt-3">
            <Note>
              This scores recovery of the anomaly shapes that were injected — flares, eclipses, step
              changes, noise bursts. Winning here proves sensitivity to those shapes, not to
              phenomena nobody thought to inject, which is the actual discovery case.
            </Note>
            <Table head={["Method", "ROC-AUC", "Avg precision", "P@k", "R@k", "Seconds"]}>
              {comparison.methods.map((method) => (
                <tr
                  key={method.name}
                  className={`border-b border-[var(--color-edge)]/50 ${
                    method.name === comparison.best_method ? "text-[var(--color-accent)]" : ""
                  }`}
                >
                  <td className="px-2 py-1.5 font-mono">{method.name}</td>
                  <td className="px-2 py-1.5">{num(method.roc_auc, 4)}</td>
                  <td className="px-2 py-1.5">{num(method.average_precision, 4)}</td>
                  <td className="px-2 py-1.5">{num(method.precision_at_k, 4)}</td>
                  <td className="px-2 py-1.5">{num(method.recall_at_k, 4)}</td>
                  <td className="px-2 py-1.5">{num(method.seconds, 1)}</td>
                </tr>
              ))}
            </Table>
          </div>
        )}
      </Panel>

      {sweep && (
        <Panel
          icon={SlidersHorizontal}
          title={`Hyperparameter sweep · ${sweep.kind}`}
          description={`${sweep.trials.length} configurations over ${sweep.rows} sequences, seeds ${sweep.seeds.join(", ")}`}
        >
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Badge tone={sweep.separated ? "ok" : "warn"}>
              {sweep.separated ? "winner separated" : "undecided"}
            </Badge>
            {sweep.best && (
              <span className="font-mono text-xs text-[var(--color-accent)]">
                {JSON.stringify(sweep.best)}
              </span>
            )}
          </div>
          {!sweep.separated && <Note tone="warn">{sweep.note}</Note>}
          <Table head={["Configuration", "Params", "ROC-AUC", "95% seed interval", "Avg precision", "Seeds"]}>
            {sweep.trials.map((trial, index) => (
              <tr key={index} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5 font-mono text-[10px]">
                  {Object.entries(trial.parameters)
                    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
                    .join(" ")}
                </td>
                <td className="px-2 py-1.5">{trial.model_parameters.toLocaleString()}</td>
                <td className="px-2 py-1.5">{num(trial.roc_auc?.mean, 4)}</td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">
                  {trial.roc_auc
                    ? `[${num(trial.roc_auc.ci95[0], 3)}, ${num(trial.roc_auc.ci95[1], 3)}]`
                    : "—"}
                </td>
                <td className="px-2 py-1.5">{num(trial.average_precision?.mean, 4)}</td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">
                  {trial.scored_seeds}
                  {trial.note && ` · ${trial.note}`}
                </td>
              </tr>
            ))}
          </Table>
        </Panel>
      )}

      <Panel icon={Boxes} title="Library versions" description="What this engine actually loaded.">
        {versions.data ? (
          <KeyValue rows={Object.entries(versions.data).map(([k, v]) => [k, v])} />
        ) : (
          <Empty>{versions.error ?? "Reading versions…"}</Empty>
        )}
      </Panel>
    </div>
  );
}

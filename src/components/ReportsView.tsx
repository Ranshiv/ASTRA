/** Research reports: exports, dataset manifests and label-based review.
 *
 * The review metrics are deliberately gated. Precision and recall computed
 * from a handful of labels look like results and are not, so when the sample
 * is too small the engine reports the shortfall instead of a number, and this
 * view shows the shortfall rather than hiding an empty table.
 */
import { BarChart3, Copy, Download, FileText, ListChecks, Package, Rss, ShieldCheck } from "lucide-react";
import { useState } from "react";

import {
  engine,
  type BroadcastFeedResult,
  type ReproducibilityBundle,
  type ReproducibilityBundleVerification,
  type ResearchBenchmarkRunResult,
  type ReviewEvaluation,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Select, Table, num, useAction, useAsync } from "@/components/ui";

/** `research.benchmark.run`'s per-result `notes` field is
 * `"method=<name>; ..."` (see `research/benchmark.py: run_cross_survey_anomaly`)
 * -- the only place the method name lives, since `ResearchResultRecord`
 * itself has no dedicated field for it. */
function methodFromNotes(notes: string): string {
  return /method=([^;]+)/.exec(notes)?.[1]?.trim() ?? "?";
}

const FORMATS = ["csv", "fits", "pdf"] as const;

export function ReportsView({ projectId }: { projectId?: string }) {
  const manifests = useAsync(() => engine.manifests(projectId), [projectId]);
  const labels = useAsync(() => engine.labelSummary(projectId), [projectId]);
  const candidates = useAsync(() => engine.candidates("default", 1, projectId), [projectId]);
  const matrices = useAsync(() => engine.featuresList(projectId), [projectId]);
  const [exported, setExported] = useState<Array<{ format: string; path: string; count: number }>>([]);
  const [feedThreshold, setFeedThreshold] = useState("0.5");
  const [feed, setFeed] = useState<BroadcastFeedResult | null>(null);
  const [review, setReview] = useState<ReviewEvaluation | null>(null);
  const [bundles, setBundles] = useState<Record<string, ReproducibilityBundle>>({});
  const [verifications, setVerifications] = useState<Record<string, ReproducibilityBundleVerification>>({});
  const [benchmarkDatasetId, setBenchmarkDatasetId] = useState("");
  const [benchmarkMatrixName, setBenchmarkMatrixName] = useState("");
  const [benchmarkId, setBenchmarkId] = useState("cross-survey-anomaly-demo");
  const [benchmarkSplitId, setBenchmarkSplitId] = useState("");
  const [injectionFraction, setInjectionFraction] = useState("0.1");
  const [benchmarkRun, setBenchmarkRun] = useState<ResearchBenchmarkRunResult | null>(null);
  const action = useAction();

  async function buildBundle(datasetId: string) {
    await action.run(`Building reproducibility bundle for ${datasetId}…`, async () => {
      try {
        const bundle = await engine.researchBundleBuild(datasetId, [], projectId);
        setBundles((current) => ({ ...current, [datasetId]: bundle }));
        setVerifications((current) => {
          const { [datasetId]: _drop, ...rest } = current;
          return rest;
        });
        return `Bundle built for ${datasetId}.`;
      } catch (err) {
        // A manifest that vanished from disk since this list was last read
        // (moved, deleted outside ASTRA) otherwise leaves a permanently
        // failing row with no way to clear it -- reload so the table
        // reflects what's actually there, then let the error still surface.
        void manifests.reload();
        throw err;
      }
    });
  }

  async function verifyBundle(datasetId: string) {
    await action.run(`Verifying bundle for ${datasetId}…`, async () => {
      const result = await engine.researchBundleVerify(datasetId, projectId);
      setVerifications((current) => ({ ...current, [datasetId]: result }));
      return result.valid ? `Bundle for ${datasetId} verified.` : `Bundle for ${datasetId} failed verification.`;
    });
  }

  async function runBenchmark() {
    const datasetId = benchmarkDatasetId || manifests.data?.[0]?.dataset_id;
    const matrixName = benchmarkMatrixName || matrices.data?.[0]?.name;
    const splitId = benchmarkSplitId || (datasetId ? `${datasetId}_object_split` : "");
    if (!datasetId || !matrixName || !benchmarkId || !splitId) {
      action.setStatus("Dataset, matrix, benchmark ID and split ID are all required.");
      return;
    }
    await action.run(`Running benchmark ${benchmarkId} on ${matrixName}…`, async () => {
      const result = await engine.researchBenchmarkRun(
        matrixName, benchmarkId, splitId, datasetId,
        Number(injectionFraction) || 0.1, projectId,
      );
      setBenchmarkRun(result);
      const dropped = result.dropped_out_of_manifest_rows;
      return `${result.results.length} results over ${result.matrix_rows_scored} rows` +
        (dropped > 0 ? ` (${dropped} out-of-manifest rows dropped).` : ".");
    });
  }

  async function generateFeed() {
    await action.run("Writing local feed file…", async () => {
      const result = await engine.broadcastFeed("default", Number(feedThreshold) || 0.5, projectId);
      setFeed(result);
      return `${result.count} candidates written to the local feed file.`;
    });
  }

  async function copyText(value: string, label: string) {
    try {
      await navigator.clipboard?.writeText(value);
      action.setStatus(`${label} copied.`);
    } catch {
      action.setStatus(`Could not copy ${label.toLowerCase()}.`);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={FileText}
        title="Export candidate run"
        description={
          candidates.data
            ? `${candidates.data.count} candidates in the current run`
            : "No candidate run loaded."
        }
        actions={FORMATS.map((format) => (
          <Button
            key={format}
            icon={Download}
            disabled={action.busy}
            onClick={() =>
              void action.run(`Exporting ${format.toUpperCase()}…`, async () => {
                const result = await engine.exportCandidates(format, "default", projectId);
                setExported((current) => [
                  { format, path: result.path, count: result.count },
                  ...current.filter((item) => item.format !== format),
                ]);
                return `${result.count} candidates written as ${format.toUpperCase()}.`;
              })
            }
          >
            {format.toUpperCase()}
          </Button>
        ))}
      >
        {action.status && <Note>{action.status}</Note>}
        {exported.length === 0 ? (
          <Empty>Exports are written into the project's reports directory.</Empty>
        ) : (
          <Table caption="Candidate exports" head={["Format", "Candidates", "Path"]}>
            {exported.map((item) => (
              <tr key={item.format} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5 uppercase">{item.format}</td>
                <td className="px-2 py-1.5">{item.count}</td>
                <td className="px-2 py-1.5 break-all font-mono text-[var(--color-accent)]">
                  <div className="flex items-center gap-2">
                    <span>{item.path}</span>
                    <button type="button" onClick={() => void copyText(item.path, "Export path")} aria-label={`Copy ${item.format.toUpperCase()} export path`} className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"><Copy size={12} /></button>
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        )}

        <div className="mt-3 border-t border-[var(--color-edge)]/50 pt-3">
          <p className="mb-1.5 text-[11px] font-medium text-[var(--color-muted)]">
            Local feed file — not published anywhere; share the path yourself.
          </p>
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Score threshold" value={feedThreshold} onChange={setFeedThreshold} width="w-24" />
            <Button icon={Rss} disabled={action.busy} onClick={() => void generateFeed()}>
              Generate local feed file
            </Button>
          </div>
          {feed && (
            <div className="mt-2 flex items-center gap-2 font-mono text-xs text-[var(--color-accent)]">
              <span className="break-all">{feed.path}</span>
              <button type="button" onClick={() => void copyText(feed.path, "Feed path")} aria-label="Copy feed file path" className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"><Copy size={12} /></button>
              <Badge tone="muted">{feed.count} candidates ≥ {feed.threshold}</Badge>
            </div>
          )}
        </div>
      </Panel>

      <Panel
        icon={ListChecks}
        title="Human review"
        description="Check whether the saved labels are sufficient for trustworthy evaluation metrics."
        actions={
          <Button
            onClick={() =>
              void action.run("Evaluating against recorded labels…", async () => {
                const result = await engine.candidatesEvaluate("default", projectId);
                setReview(result);
                return result.ready
                  ? `ROC-AUC ${num(result.roc_auc, 4)} over ${result.labels} labels.`
                  : (result.reason ?? "Not enough labels.");
              })
            }
          >
            Evaluate
          </Button>
        }
      >
        {labels.data && (
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            <Badge tone="muted">{labels.data.total} labelled</Badge>
            {Object.entries(labels.data.by_label).map(([label, count]) => (
              <Badge key={label} tone={count > 0 ? "accent" : "muted"}>
                {label.replace(/_/g, " ")}: {count}
              </Badge>
            ))}
          </div>
        )}
        {review && !review.ready && (
          <div className="rounded border border-[var(--color-warn)]/40 bg-[var(--color-warn)]/5 p-2">
            <Note tone="warn">
              {review.reason ?? "Insufficient labels."} Metrics need at least{" "}
              {review.minimum_labels} labels with {review.minimum_per_class} per class; there are{" "}
              {review.labels} ({review.positives} positive, {review.negatives} negative). Reporting
              a precision from fewer would look like a result without being one.
            </Note>
          </div>
        )}
        {review?.ready && (
          <KeyValue
            rows={[
              ["Labels", `${review.labels} (${review.positives}+ / ${review.negatives}−)`],
              ["Threshold", num(review.threshold, 2)],
              ["Precision", num(review.precision, 4)],
              ["Recall", num(review.recall, 4)],
              ["F1", num(review.f1, 4)],
              ["ROC-AUC", num(review.roc_auc, 4)],
              ["Average precision", num(review.average_precision, 4)],
            ]}
          />
        )}
      </Panel>

      <Panel
        icon={Package}
        title="Dataset manifests"
        description="What was acquired, when, and its content hash."
        actions={<Button onClick={() => void manifests.reload()}>Refresh</Button>}
      >
        {manifests.error && <Note tone="bad">{manifests.error}</Note>}
        {(manifests.data?.length ?? 0) === 0 ? (
          <Empty>No acquisitions recorded yet.</Empty>
        ) : (
          <Table caption="Acquired dataset manifests" head={["Dataset", "Acquired", "Surveys", "Objects", "Content hash", "Reproducibility bundle"]}>
            {manifests.data!.map((item) => {
              const bundle = bundles[item.dataset_id];
              const verification = verifications[item.dataset_id];
              return (
                <tr key={item.dataset_id} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{item.dataset_id}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">{item.created_utc}</td>
                  <td className="px-2 py-1.5">{item.surveys.join(", ")}</td>
                  <td className="px-2 py-1.5">{item.objects.toLocaleString()}</td>
                  <td className="px-2 py-1.5 font-mono text-[var(--color-muted)]">
                    <div className="flex items-center gap-2">
                      <span title={item.content_hash}>{item.content_hash?.slice(0, 16)}…</span>
                      {item.content_hash && <button type="button" onClick={() => void copyText(item.content_hash!, "Content hash")} aria-label={`Copy full hash for ${item.dataset_id}`} className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"><Copy size={12} /></button>}
                    </div>
                  </td>
                  <td className="px-2 py-1.5">
                    {!bundle ? (
                      <Button
                        icon={ShieldCheck}
                        disabled={action.busy || !item.content_hash}
                        onClick={() => void buildBundle(item.dataset_id)}
                      >
                        {item.content_hash ? "Build bundle" : "Not sealed"}
                      </Button>
                    ) : (
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 font-mono text-[var(--color-muted)]">
                          <span title={bundle.bundle_hash}>{bundle.bundle_hash.slice(0, 16)}…</span>
                          <button type="button" onClick={() => void copyText(bundle.bundle_hash, "Bundle hash")} aria-label={`Copy bundle hash for ${item.dataset_id}`} className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"><Copy size={12} /></button>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <Badge tone={bundle.signature_hex ? "accent" : "muted"}>
                            {bundle.signature_hex ? "signed" : "unsigned"}
                          </Badge>
                          {verification && (
                            <Badge tone={verification.valid ? "ok" : "bad"}>
                              {verification.valid ? "verified" : "invalid"}
                            </Badge>
                          )}
                          <Button disabled={action.busy} onClick={() => void verifyBundle(item.dataset_id)}>
                            Verify
                          </Button>
                        </div>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </Table>
        )}
      </Panel>

      <Panel
        icon={BarChart3}
        title="Research benchmark"
        description="Score the cross-survey anomaly baselines against a saved feature matrix, bound to a dataset manifest and split."
      >
        <div className="flex flex-wrap items-end gap-3">
          <Select
            label="Dataset"
            value={benchmarkDatasetId || manifests.data?.[0]?.dataset_id || ""}
            onChange={setBenchmarkDatasetId}
            options={(manifests.data ?? []).map((m) => ({ value: m.dataset_id, label: m.dataset_id }))}
          />
          <Select
            label="Feature matrix"
            value={benchmarkMatrixName || matrices.data?.[0]?.name || ""}
            onChange={setBenchmarkMatrixName}
            options={(matrices.data ?? []).map((m) => ({ value: m.name, label: `${m.name} (${m.rows} rows)` }))}
          />
          <Field label="Benchmark ID" value={benchmarkId} onChange={setBenchmarkId} width="w-48" />
          <Field
            label="Split ID"
            value={benchmarkSplitId}
            onChange={setBenchmarkSplitId}
            placeholder={benchmarkDatasetId ? `${benchmarkDatasetId}_object_split` : "dataset_object_split"}
            width="w-56"
          />
          <Field label="Injection fraction" value={injectionFraction} onChange={setInjectionFraction} width="w-24" />
          <Button icon={BarChart3} disabled={action.busy} onClick={() => void runBenchmark()}>
            Run benchmark
          </Button>
        </div>
        {action.status && <Note>{action.status}</Note>}
        {!benchmarkRun ? (
          <Empty>No benchmark run yet. Slow: scores five baselines at every declared seed.</Empty>
        ) : (
          <>
            <div className="my-2 flex flex-wrap items-center gap-1.5">
              <Badge tone="muted">experiment {benchmarkRun.experiment_id}</Badge>
              <Badge tone="muted">split {benchmarkRun.split_id}</Badge>
              <Badge tone="accent">{benchmarkRun.matrix_rows_scored} rows scored</Badge>
              {benchmarkRun.dropped_out_of_manifest_rows > 0 && (
                <Badge tone="warn">{benchmarkRun.dropped_out_of_manifest_rows} out-of-manifest rows dropped</Badge>
              )}
            </div>
            <Table caption="Benchmark results" head={["Method", "Metric", "Value", "95% CI", "Seed", "Label"]}>
              {benchmarkRun.results.map((result, index) => (
                <tr key={`${result.seed}-${index}`} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{methodFromNotes(result.notes)}</td>
                  <td className="px-2 py-1.5">{result.metric}</td>
                  <td className="px-2 py-1.5">{num(result.value, 4)}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">
                    [{num(result.confidence_interval[0], 4)}, {num(result.confidence_interval[1], 4)}]
                  </td>
                  <td className="px-2 py-1.5">{result.seed}</td>
                  <td className="px-2 py-1.5">
                    <Badge tone={result.synthetic ? "warn" : "ok"}>
                      {result.synthetic ? "synthetic label" : "real label"}
                    </Badge>
                  </td>
                </tr>
              ))}
            </Table>
          </>
        )}
      </Panel>
    </div>
  );
}

/** Research reports: exports, dataset manifests and label-based review.
 *
 * The review metrics are deliberately gated. Precision and recall computed
 * from a handful of labels look like results and are not, so when the sample
 * is too small the engine reports the shortfall instead of a number, and this
 * view shows the shortfall rather than hiding an empty table.
 */
import { Copy, Download, FileText, ListChecks, Package } from "lucide-react";
import { useState } from "react";

import { engine, type ReviewEvaluation } from "@/lib/engine";
import { Badge, Button, Empty, KeyValue, Note, Panel, Table, num, useAction, useAsync } from "@/components/ui";

const FORMATS = ["csv", "fits", "pdf"] as const;

export function ReportsView({ projectId }: { projectId?: string }) {
  const manifests = useAsync(() => engine.manifests(projectId), [projectId]);
  const labels = useAsync(() => engine.labelSummary(projectId), [projectId]);
  const candidates = useAsync(() => engine.candidates("default", 1, projectId), [projectId]);
  const [exported, setExported] = useState<Array<{ format: string; path: string; count: number }>>([]);
  const [review, setReview] = useState<ReviewEvaluation | null>(null);
  const action = useAction();

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
                    <button type="button" onClick={() => void copyText(item.path, "Export path")} aria-label={`Copy ${item.format.toUpperCase()} export path`} className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)]"><Copy size={12} /></button>
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        )}
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
          <Table caption="Acquired dataset manifests" head={["Dataset", "Acquired", "Surveys", "Objects", "Content hash"]}>
            {manifests.data!.map((item) => (
              <tr key={item.dataset_id} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5 font-mono">{item.dataset_id}</td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">{item.created_utc}</td>
                <td className="px-2 py-1.5">{item.surveys.join(", ")}</td>
                <td className="px-2 py-1.5">{item.objects.toLocaleString()}</td>
                <td className="px-2 py-1.5 font-mono text-[var(--color-muted)]">
                  <div className="flex items-center gap-2">
                    <span title={item.content_hash}>{item.content_hash?.slice(0, 16)}…</span>
                    {item.content_hash && <button type="button" onClick={() => void copyText(item.content_hash!, "Content hash")} aria-label={`Copy full hash for ${item.dataset_id}`} className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel-2)] hover:text-[var(--color-text)]"><Copy size={12} /></button>}
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        )}
      </Panel>
    </div>
  );
}

/** Plan section 41's landing screen.
 *
 * The three actions the plan names — New Research Project, Explore Existing
 * Data, Investigate Candidates — plus enough state to answer "is this
 * installation ready, and what is in it" without clicking into four views.
 */
import {
  Activity,
  Compass,
  Cpu,
  Database,
  FlaskConical,
  FolderPlus,
  HardDrive,
  ListChecks,
  Package,
  Telescope,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { SectionId } from "@/components/Sidebar";
import { engine } from "@/lib/engine";
import { Badge, Button, Empty, Note, Panel, Table, num, useAsync } from "@/components/ui";

function Stat({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-3">
      <div className="flex items-center gap-2 text-[var(--color-muted)]">
        <Icon size={14} strokeWidth={2} />
        <span className="text-xs">{label}</span>
      </div>
      <p className="mt-1.5 truncate text-lg font-semibold" title={value}>
        {value}
      </p>
      {detail && <p className="mt-0.5 truncate text-[11px] text-[var(--color-muted)]">{detail}</p>}
    </div>
  );
}

export function Dashboard({
  onNavigate,
  projectId,
}: {
  onNavigate: (section: SectionId) => void;
  projectId?: string;
}) {
  const hardware = useAsync(() => engine.hardware());
  const usage = useAsync(() => engine.storeUsage());
  const labels = useAsync(() => engine.labelSummary(projectId), [projectId]);
  const experiments = useAsync(() => engine.experiments(projectId), [projectId]);
  const manifests = useAsync(() => engine.manifests(projectId), [projectId]);
  const candidates = useAsync(() => engine.candidates("default", 5, projectId), [projectId]);

  const dataset = usage.data?.dataset;
  const totalCurves = Object.values(usage.data?.surveys ?? {}).reduce(
    (sum, entry) => sum + entry.curves,
    0,
  );

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Telescope}
        title="ASTRA"
        description="Astronomical Survey & Transient Research Analyzer"
        actions={
          <>
            <Button icon={FolderPlus} tone="accent" onClick={() => onNavigate("projects")}>
              New research project
            </Button>
            <Button icon={Compass} onClick={() => onNavigate("sky")}>
              Explore existing data
            </Button>
            <Button icon={ListChecks} onClick={() => onNavigate("candidates")}>
              Investigate candidates
            </Button>
          </>
        }
      >
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Stat
            icon={Cpu}
            label="Execution mode"
            value={hardware.data ? hardware.data.device.toUpperCase() : "…"}
            detail={hardware.data?.gpu?.name ?? hardware.data?.reason}
          />
          <Stat
            icon={Activity}
            label="Stored light curves"
            value={usage.data ? totalCurves.toLocaleString() : "…"}
            detail={
              usage.data
                ? Object.entries(usage.data.surveys)
                    .map(([survey, entry]) => `${survey} ${entry.curves}`)
                    .join(" · ")
                : undefined
            }
          />
          <Stat
            icon={HardDrive}
            label="Dataset usage"
            value={dataset ? `${num(dataset.used_gb, 2)} GB` : "…"}
            detail={dataset ? `of ${num(dataset.cap_gb, 0)} GB cap` : undefined}
          />
          <Stat
            icon={ListChecks}
            label="Candidates"
            value={candidates.data ? candidates.data.count.toLocaleString() : "…"}
            detail={labels.data ? `${labels.data.total} labelled` : undefined}
          />
        </div>

        {hardware.data && !hardware.data.torch_available && (
          <Note>
            PyTorch is not present, so this engine runs in CPU mode. Acquisition, features,
            baseline anomaly detection, cross-survey matching, ranking and export all work
            normally; only the deep models are unavailable.
          </Note>
        )}
        {hardware.error && <Note tone="bad">{hardware.error}</Note>}
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel
          icon={ListChecks}
          title="Top candidates"
          description="Highest-ranked in the current run."
          actions={<Button onClick={() => onNavigate("candidates")}>Open workspace</Button>}
        >
          {(candidates.data?.candidates.length ?? 0) === 0 ? (
            <Empty>
              {candidates.error
                ? "No candidate run yet — build one in the candidate workspace."
                : "No candidates in the current run."}
            </Empty>
          ) : (
            <Table head={["Rank", "Candidate", "Survey", "Score"]}>
              {candidates.data!.candidates.map((candidate) => (
                <tr key={candidate.candidate_id} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5">#{candidate.rank}</td>
                  <td className="px-2 py-1.5 font-mono">{candidate.candidate_id}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">
                    {candidate.survey} · {candidate.band}
                  </td>
                  <td className="px-2 py-1.5">
                    {num(candidate.score.supervised_probability ?? candidate.score.total, 3)}
                  </td>
                </tr>
              ))}
            </Table>
          )}
        </Panel>

        <Panel
          icon={FlaskConical}
          title="Recent experiments"
          description="Every run is recorded with its provenance."
          actions={<Button onClick={() => onNavigate("experiments")}>Open</Button>}
        >
          {(experiments.data?.length ?? 0) === 0 ? (
            <Empty>No experiments recorded yet.</Empty>
          ) : (
            <Table head={["Experiment", "Kind", "Recorded", ""]}>
              {experiments.data!.slice(-6).reverse().map((item) => (
                <tr key={item.experiment_id} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{item.experiment_id}</td>
                  <td className="px-2 py-1.5">{item.kind}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">{item.created_utc}</td>
                  <td className="px-2 py-1.5">{item.failed && <Badge tone="bad">failed</Badge>}</td>
                </tr>
              ))}
            </Table>
          )}
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel
          icon={Package}
          title="Recent acquisitions"
          description="Dataset manifests, with content hashes."
          actions={<Button onClick={() => onNavigate("acquire")}>Acquire more</Button>}
        >
          {(manifests.data?.length ?? 0) === 0 ? (
            <Empty>Nothing acquired yet.</Empty>
          ) : (
            <Table head={["Dataset", "Surveys", "Objects", "Acquired"]}>
              {manifests.data!.slice(-6).reverse().map((item) => (
                <tr key={item.dataset_id} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono">{item.dataset_id}</td>
                  <td className="px-2 py-1.5">{item.surveys.join(", ")}</td>
                  <td className="px-2 py-1.5">{item.objects.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-[var(--color-muted)]">{item.created_utc}</td>
                </tr>
              ))}
            </Table>
          )}
        </Panel>

        <Panel
          icon={Database}
          title="Human review"
          description="Labels feed the supervised ranker (plan section 22)."
          actions={<Button onClick={() => onNavigate("reports")}>Reports</Button>}
        >
          {labels.data ? (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(labels.data.by_label).map(([label, count]) => (
                <Badge key={label} tone={count > 0 ? "accent" : "muted"}>
                  {label.replace(/_/g, " ")}: {count}
                </Badge>
              ))}
            </div>
          ) : (
            <Empty>{labels.error ?? "Reading labels…"}</Empty>
          )}
        </Panel>
      </div>
    </div>
  );
}

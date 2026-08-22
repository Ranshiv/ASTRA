import { lazy, Suspense, useEffect, useState } from "react";
import { ArrowRight, Check, Circle, FolderKanban } from "lucide-react";

import { AcquirePanel } from "@/components/AcquirePanel";
import { Empty } from "@/components/ui";
import { CurveExplorer } from "@/components/CurveExplorer";
import { CrossSurveyPanel } from "@/components/CrossSurveyPanel";
import { Dashboard } from "@/components/Dashboard";
import { ExperimentsView } from "@/components/ExperimentsView";
import { ExplainPanel } from "@/components/ExplainPanel";
import { EventsView } from "@/components/EventsView";
import { ModelsView } from "@/components/ModelsView";
import { ProjectWorkspace } from "@/components/ProjectWorkspace";
import { ReportsView } from "@/components/ReportsView";
import { Sidebar, type SectionId } from "@/components/Sidebar";
import { SettingsView } from "@/components/SettingsView";
import { SkyExplorer } from "@/components/SkyExplorer";
import { StatusStrip } from "@/components/StatusPanel";
import { TitleBar } from "@/components/TitleBar";
const CandidateWorkspace = lazy(() => import("@/components/CandidateWorkspace").then((module) => ({ default: module.CandidateWorkspace })));
import {
  engine,
  type CacheStatus,
  type DeviceReport,
  type EnginePaths,
  type SurveyInfo,
  type ResearchProject,
} from "@/lib/engine";
import { useHashNavigation } from "@/lib/navigation";

const PAGE_TITLES: Record<SectionId, string> = {
  dashboard: "ASTRA",
  projects: "Projects",
  acquire: "Acquire observations",
  events: "Event inbox",
  sky: "Sky explorer",
  curves: "Light curves",
  candidates: "Candidate workspace",
  crosssurvey: "Cross-survey evidence",
  explain: "Why a candidate scored",
  experiments: "Experiments",
  models: "Models",
  reports: "Research reports",
  settings: "Settings",
};

const WORKFLOW: Array<{ id: SectionId; label: string; description: string }> = [
  { id: "projects", label: "Set up", description: "Choose a project and survey coverage" },
  { id: "acquire", label: "Acquire", description: "Bring observations into the project" },
  { id: "candidates", label: "Analyze", description: "Generate and rank candidates" },
  { id: "explain", label: "Review", description: "Check evidence and artifact risk" },
  { id: "reports", label: "Report", description: "Export reproducible results" },
];

const SECTION_IDS = Object.keys(PAGE_TITLES) as SectionId[];

/** Sections that need an active project to be useful, matching WorkflowRail's
 * own gating so Sidebar doesn't offer a path WorkflowRail blocks. */
const PROJECT_GATED_SECTIONS: ReadonlySet<SectionId> = new Set(
  WORKFLOW.slice(1).map((item) => item.id),
);

function WorkflowRail({ section, activeProject, onNavigate }: {
  section: SectionId;
  activeProject: ResearchProject | null;
  onNavigate: (section: SectionId) => void;
}) {
  const currentIndex = WORKFLOW.findIndex((item) => item.id === section);
  return (
    <nav aria-label="Research workflow" className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <FolderKanban size={16} className="shrink-0 text-[var(--color-accent)]" />
          <span className="truncate text-sm font-medium">
            {activeProject?.name ?? "No project selected"}
          </span>
        </div>
        {!activeProject && (
          <button type="button" onClick={() => onNavigate("projects")} className="inline-flex min-h-9 shrink-0 items-center gap-1.5 rounded border border-[var(--color-accent)] px-3 py-1.5 text-xs text-[var(--color-accent)] hover:bg-[var(--color-accent)]/10">
            Create project <ArrowRight size={12} />
          </button>
        )}
      </div>
      <ol className="mt-3 grid gap-2 sm:grid-cols-5">
        {WORKFLOW.map((item, index) => {
          const complete = Boolean(activeProject) && currentIndex > index;
          const active = item.id === section || (section === "dashboard" && index === 0);
          const available = Boolean(activeProject) || index === 0;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => available && onNavigate(item.id)}
                disabled={!available}
                aria-current={active ? "step" : undefined}
                title={item.description}
                className={`flex min-h-11 w-full items-center gap-2 rounded border px-2.5 py-2 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] ${active ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]" : "border-[var(--color-edge)] text-[var(--color-muted)] hover:border-[var(--color-muted)]"} disabled:cursor-not-allowed disabled:opacity-45`}
              >
                {complete ? <Check size={14} className="shrink-0 text-[var(--color-ok)]" /> : <Circle size={14} className="shrink-0" />}
                <span className="min-w-0 truncate">{item.label}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default function App() {
  const [device, setDevice] = useState<DeviceReport | null>(null);
  const [cache, setCache] = useState<CacheStatus | null>(null);
  const [paths, setPaths] = useState<EnginePaths | null>(null);
  const [surveys, setSurveys] = useState<SurveyInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [curvesCount, setCurvesCount] = useState<number | undefined>(undefined);
  const [candidatesCount, setCandidatesCount] = useState<number | undefined>(undefined);
  const { section, navigate } = useHashNavigation<SectionId>(SECTION_IDS, "dashboard");
  const [activeProject, setActiveProject] = useState<ResearchProject | null>(null);
  const [environmentRefresh, setEnvironmentRefresh] = useState(0);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [hardware, cacheStatus, enginePaths, surveyList, projectList] = await Promise.all([
          engine.hardware(),
          engine.cacheStatus(),
          engine.paths(),
          engine.surveys(),
          engine.projects(false),
        ]);
        if (cancelled) return;
        setDevice(hardware);
        setCache(cacheStatus);
        setPaths(enginePaths);
        setSurveys(surveyList);
        setActiveProject(projectList.find((project) => project.status === "active") ?? projectList[0] ?? null);
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    })();

    engine.curvesList(undefined, 500, activeProject?.project_id).then((curves) => {
      if (!cancelled) setCurvesCount(curves.length);
    }).catch(() => {});
    engine.candidates().then((result) => {
      if (!cancelled) setCandidatesCount(result.count);
    }).catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [activeProject?.project_id, environmentRefresh]);

  return (
    <div className="flex h-screen flex-col">
      <TitleBar />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <Sidebar
          active={section}
          onSelect={navigate}
          dataRoot={activeProject?.data_root ?? paths?.root}
          gatedSections={PROJECT_GATED_SECTIONS}
          hasProject={Boolean(activeProject)}
        />

        <div className="flex-1 overflow-y-auto">
          <main id="main-content" className="mx-auto flex max-w-[1400px] flex-col gap-5 px-4 pb-8 pt-4 sm:px-6 lg:px-8">
            <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:rounded focus:bg-[var(--color-panel)] focus:px-3 focus:py-2">Skip to main content</a>
            <div className="sticky top-0 z-10 flex flex-col gap-4 border-b border-[var(--color-edge)] bg-[var(--color-void)]/95 pb-4 pt-2 backdrop-blur">
              <header className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h1 className="text-balance text-2xl font-semibold tracking-tight">{PAGE_TITLES[section]}</h1>
                {section === "dashboard" && (
                  <p className="mt-1 text-sm text-[var(--color-muted)]">
                    Astronomical Survey &amp; Transient Research Analyzer
                  </p>
                )}
                </div>
                {activeProject && <p className="max-w-[40%] truncate rounded border border-[var(--color-edge)] px-2.5 py-1.5 text-xs text-[var(--color-muted)]" title={activeProject.data_root}>Project: <span className="text-[var(--color-text)]">{activeProject.name}</span></p>}
              </header>
              <WorkflowRail section={section} activeProject={activeProject} onNavigate={navigate} />
              <StatusStrip device={device} cache={cache} error={error} curvesCount={curvesCount} candidatesCount={candidatesCount} onRetry={() => { setError(null); setEnvironmentRefresh((value) => value + 1); }} />
            </div>

            {section === "dashboard" && <Dashboard onNavigate={navigate} projectId={activeProject?.project_id} />}

            {section === "projects" && <ProjectWorkspace activeProject={activeProject} surveys={surveys} onSelect={setActiveProject} />}

            {section === "acquire" && (
              surveys.length > 0 ? (
                <AcquirePanel
                  surveys={surveys}
                  projectId={activeProject?.project_id}
                  queryRegions={activeProject?.query_regions}
                />
              ) : (
                <Empty>No surveys are available yet. Check the engine connection or try again once catalogs finish loading.</Empty>
              )
            )}

            {section === "events" && <EventsView projectId={activeProject?.project_id} />}

            {section === "curves" && <CurveExplorer projectId={activeProject?.project_id} />}

            {section === "sky" && <SkyExplorer projectId={activeProject?.project_id} />}

            {section === "candidates" && (
              <Suspense fallback={<p className="text-xs text-[var(--color-muted)]">Loading candidate tools…</p>}>
                <CandidateWorkspace projectId={activeProject?.project_id} />
              </Suspense>
            )}

            {section === "crosssurvey" && <CrossSurveyPanel projectId={activeProject?.project_id} />}
            {section === "explain" && <ExplainPanel projectId={activeProject?.project_id} />}
            {section === "experiments" && <ExperimentsView projectId={activeProject?.project_id} />}
            {section === "models" && <ModelsView projectId={activeProject?.project_id} />}
            {section === "reports" && <ReportsView projectId={activeProject?.project_id} />}
            {section === "settings" && <SettingsView />}
          </main>
        </div>
      </div>
    </div>
  );
}

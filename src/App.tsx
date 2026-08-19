import { lazy, Suspense, useEffect, useState } from "react";

import { AcquirePanel } from "@/components/AcquirePanel";
import { CurveExplorer } from "@/components/CurveExplorer";
import { CrossSurveyPanel } from "@/components/CrossSurveyPanel";
import { Dashboard } from "@/components/Dashboard";
import { ExperimentsView } from "@/components/ExperimentsView";
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

const PAGE_TITLES: Record<SectionId, string> = {
  dashboard: "ASTRA",
  projects: "Projects",
  acquire: "Acquire observations",
  sky: "Sky explorer",
  curves: "Light curves",
  candidates: "Candidate workspace",
  crosssurvey: "Cross-survey evidence",
  experiments: "Experiments",
  models: "Models",
  reports: "Research reports",
  settings: "Settings",
};

export default function App() {
  const [device, setDevice] = useState<DeviceReport | null>(null);
  const [cache, setCache] = useState<CacheStatus | null>(null);
  const [paths, setPaths] = useState<EnginePaths | null>(null);
  const [surveys, setSurveys] = useState<SurveyInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [curvesCount, setCurvesCount] = useState<number | undefined>(undefined);
  const [candidatesCount, setCandidatesCount] = useState<number | undefined>(undefined);
  const [section, setSection] = useState<SectionId>("dashboard");
  const [activeProject, setActiveProject] = useState<ResearchProject | null>(null);

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
  }, [activeProject?.project_id]);

  return (
    <div className="flex h-screen flex-col">
      <TitleBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active={section} onSelect={setSection} dataRoot={activeProject?.data_root ?? paths?.root} />

        <div className="flex-1 overflow-y-auto">
          <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-8 pb-8">
            <div className="sticky top-0 z-10 flex flex-col gap-6 border-b border-[var(--color-edge)] bg-[var(--color-void)] pb-4 pt-8">
              <header>
                <h1 className="text-2xl font-semibold tracking-tight">{PAGE_TITLES[section]}</h1>
                {section === "dashboard" && (
                  <p className="mt-1 text-sm text-[var(--color-muted)]">
                    Astronomical Survey &amp; Transient Research Analyzer
                  </p>
                )}
              </header>

              <StatusStrip
                device={device}
                cache={cache}
                error={error}
                curvesCount={curvesCount}
                candidatesCount={candidatesCount}
              />
            </div>

            {section === "dashboard" && <Dashboard onNavigate={setSection} projectId={activeProject?.project_id} />}

            {section === "projects" && <ProjectWorkspace activeProject={activeProject} surveys={surveys} onSelect={setActiveProject} />}

            {section === "acquire" && surveys.length > 0 && <AcquirePanel surveys={surveys} projectId={activeProject?.project_id} />}

            {section === "curves" && <CurveExplorer projectId={activeProject?.project_id} />}

            {section === "sky" && <SkyExplorer projectId={activeProject?.project_id} />}

            {section === "candidates" && (
              <Suspense fallback={<p className="text-xs text-[var(--color-muted)]">Loading candidate tools…</p>}>
                <CandidateWorkspace projectId={activeProject?.project_id} />
              </Suspense>
            )}

            {section === "crosssurvey" && <CrossSurveyPanel projectId={activeProject?.project_id} />}
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

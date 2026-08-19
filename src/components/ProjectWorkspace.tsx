import { Archive, CheckCircle2, FolderKanban, Plus, RefreshCw, Save } from "lucide-react";
import { useEffect, useState } from "react";

import { SectionHeader } from "@/components/SectionHeader";
import { engine, type ResearchProject, type ProjectValidation, type SurveyInfo } from "@/lib/engine";

export function ProjectWorkspace({
  activeProject,
  surveys,
  onSelect,
}: {
  activeProject: ResearchProject | null;
  surveys: SurveyInfo[];
  onSelect: (project: ResearchProject | null) => void;
}) {
  const [projects, setProjects] = useState<ResearchProject[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [selectedSurveys, setSelectedSurveys] = useState<string[]>([]);
  const [validation, setValidation] = useState<ProjectValidation | null>(null);
  const [status, setStatus] = useState("Loading projects…");
  const [busy, setBusy] = useState(false);

  async function refresh(preferredId?: string) {
    try {
      const loaded = await engine.projects(true);
      setProjects(loaded);
      const preferred = preferredId ? loaded.find((item) => item.project_id === preferredId) : undefined;
      const next = preferred ?? loaded.find((item) => item.status === "active") ?? loaded[0] ?? null;
      if (next && next.project_id !== activeProject?.project_id) onSelect(next);
      setStatus(`${loaded.length} project${loaded.length === 1 ? "" : "s"}`);
    } catch (err) {
      setStatus(String(err));
    }
  }

  useEffect(() => {
    void refresh();
    // The list is loaded once on entry; explicit refresh/create/archive actions
    // update it afterward, so selecting "New project" stays an intentional
    // empty form instead of auto-selecting an existing project.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!activeProject) return;
    setName(activeProject.name);
    setDescription(activeProject.description);
    setTags(activeProject.tags.join(", "));
    setSelectedSurveys(activeProject.selected_surveys);
    setValidation(null);
  }, [activeProject]);

  function toggleSurvey(survey: string) {
    setSelectedSurveys((current) => current.includes(survey)
      ? current.filter((item) => item !== survey)
      : [...current, survey]);
  }

  async function create() {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const project = await engine.projectCreate({
        name: name.trim(),
        description,
        tags: tags.split(",").map((item) => item.trim()).filter(Boolean),
        selectedSurveys,
      });
      setProjects((current) => [...current.filter((item) => item.project_id !== project.project_id), project]);
      onSelect(project);
      setStatus(`Created ${project.name}`);
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!activeProject || activeProject.status === "archived") return;
    setBusy(true);
    try {
      const project = await engine.projectUpdate(activeProject.project_id, {
        name: name.trim(),
        description,
        tags: tags.split(",").map((item) => item.trim()).filter(Boolean),
        selectedSurveys,
      });
      setProjects((current) => current.map((item) => item.project_id === project.project_id ? project : item));
      onSelect(project);
      setStatus("Project saved");
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function archive() {
    if (!activeProject) return;
    setBusy(true);
    try {
      const project = await engine.projectArchive(activeProject.project_id, activeProject.status !== "archived");
      setProjects((current) => current.map((item) => item.project_id === project.project_id ? project : item));
      onSelect(project);
      setStatus(project.status === "archived" ? "Project archived; it is now read-only." : "Project restored.");
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function validate() {
    if (!activeProject) return;
    setBusy(true);
    try {
      setValidation(await engine.projectValidate(activeProject.project_id));
      setStatus("Validation complete");
    } catch (err) {
      setStatus(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
      <SectionHeader
        icon={FolderKanban}
        title="Projects"
        description={`${status}. Projects keep manifests, candidate reviews, experiments, results, and reports separate.`}
        actions={
          <button type="button" onClick={() => void refresh(activeProject?.project_id)} disabled={busy} className="rounded border border-[var(--color-edge)] p-1.5 text-[var(--color-muted)] disabled:opacity-40" aria-label="Refresh projects">
            <RefreshCw size={13} />
          </button>
        }
      />

      <div className="mt-4 grid gap-4 lg:grid-cols-[16rem_1fr]">
        <div className="space-y-1 rounded border border-[var(--color-edge)] p-2">
          {projects.length === 0 && <p className="p-2 text-xs text-[var(--color-muted)]">No projects yet.</p>}
          {projects.map((project) => (
            <button
              type="button"
              key={project.project_id}
              onClick={() => onSelect(project)}
              className={`w-full rounded px-2 py-2 text-left text-xs ${activeProject?.project_id === project.project_id ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]" : "hover:bg-[var(--color-panel-2)]"}`}
            >
              <span className="block font-medium">{project.name}</span>
              <span className="mt-0.5 block font-mono text-[10px] text-[var(--color-muted)]">{project.project_id} · {project.status}</span>
            </button>
          ))}
          <button type="button" onClick={() => { onSelect(null); setName(""); setDescription(""); setTags(""); setSelectedSurveys([]); }} className="mt-2 flex w-full items-center justify-center gap-1 rounded border border-dashed border-[var(--color-edge)] px-2 py-1.5 text-xs text-[var(--color-muted)]">
            <Plus size={12} /> New project
          </button>
        </div>

        <div className="space-y-3">
          <label className="block text-xs">
            <span className="text-[var(--color-muted)]">Name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} className="mt-1 w-full rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1.5" placeholder="e.g. RR Lyrae follow-up" />
          </label>
          <label className="block text-xs">
            <span className="text-[var(--color-muted)]">Description</span>
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={3} maxLength={2000} className="mt-1 w-full resize-y rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1.5" />
          </label>
          <label className="block text-xs">
            <span className="text-[var(--color-muted)]">Tags (comma-separated)</span>
            <input value={tags} onChange={(event) => setTags(event.target.value)} className="mt-1 w-full rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1.5" />
          </label>
          <div>
            <p className="text-xs text-[var(--color-muted)]">Preferred surveys</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {surveys.map((survey) => {
                const key = survey.name.toLowerCase();
                const selected = selectedSurveys.includes(key);
                return <button type="button" key={key} onClick={() => toggleSurvey(key)} className={`rounded-full border px-2 py-1 text-xs ${selected ? "border-[var(--color-accent)] text-[var(--color-accent)]" : "border-[var(--color-edge)] text-[var(--color-muted)]"}`}>{survey.name}</button>;
              })}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {!activeProject && <button type="button" onClick={() => void create()} disabled={busy || !name.trim()} className="flex items-center gap-1 rounded border border-[var(--color-accent)] px-2.5 py-1.5 text-xs text-[var(--color-accent)] disabled:opacity-40"><Plus size={12} /> Create project</button>}
            {activeProject && <>
              <button type="button" onClick={() => void save()} disabled={busy || activeProject.status === "archived" || !name.trim()} className="flex items-center gap-1 rounded border border-[var(--color-accent)] px-2.5 py-1.5 text-xs text-[var(--color-accent)] disabled:opacity-40"><Save size={12} /> Save</button>
              <button type="button" onClick={() => void archive()} disabled={busy} className="flex items-center gap-1 rounded border border-[var(--color-edge)] px-2.5 py-1.5 text-xs disabled:opacity-40"><Archive size={12} /> {activeProject.status === "archived" ? "Restore" : "Archive"}</button>
              <button type="button" onClick={() => void validate()} disabled={busy} className="flex items-center gap-1 rounded border border-[var(--color-edge)] px-2.5 py-1.5 text-xs disabled:opacity-40"><CheckCircle2 size={12} /> Validate</button>
            </>}
          </div>
          {validation && <div className={`rounded border p-2 text-xs ${validation.valid ? "border-[var(--color-ok)]/40 text-[var(--color-ok)]" : "border-[var(--color-bad)]/40 text-[var(--color-bad)]"}`}>
            {validation.valid ? "Project layout and manifests are valid." : validation.issues.join(" ")}
          </div>}
          {activeProject && <p className="break-all font-mono text-[10px] text-[var(--color-muted)]">{activeProject.data_root}</p>}
        </div>
      </div>
    </section>
  );
}

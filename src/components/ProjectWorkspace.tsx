import { Archive, CheckCircle2, FolderKanban, Plus, RefreshCw, Save, X } from "lucide-react";
import { useEffect, useState } from "react";

import {
  engine,
  type ProjectRegion,
  type ResearchProject,
  type ProjectValidation,
  type SurveyInfo,
} from "@/lib/engine";
import { Button, Field, Note, Panel } from "@/components/ui";

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
  const [regions, setRegions] = useState<ProjectRegion[]>([]);
  const [regionRa, setRegionRa] = useState("");
  const [regionDec, setRegionDec] = useState("");
  const [regionRadius, setRegionRadius] = useState("60");
  const [validation, setValidation] = useState<ProjectValidation | null>(null);
  const [status, setStatus] = useState("Loading projects…");
  const [busy, setBusy] = useState(false);
  const [regionError, setRegionError] = useState<string | null>(null);

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
    setRegions(activeProject.query_regions ?? []);
    setValidation(null);
  }, [activeProject]);

  function toggleSurvey(survey: string) {
    setSelectedSurveys((current) => current.includes(survey)
      ? current.filter((item) => item !== survey)
      : [...current, survey]);
  }

  function addRegion() {
    const ra_deg = Number(regionRa);
    const dec_deg = Number(regionDec);
    const radius_arcsec = Number(regionRadius);
    if (!Number.isFinite(ra_deg) || ra_deg < 0 || ra_deg >= 360) {
      setRegionError("RA must be between 0 and 360 degrees.");
      return;
    }
    if (!Number.isFinite(dec_deg) || dec_deg < -90 || dec_deg > 90) {
      setRegionError("Dec must be between −90 and +90 degrees.");
      return;
    }
    if (!Number.isFinite(radius_arcsec) || radius_arcsec <= 0) {
      setRegionError("Radius must be greater than 0 arcsec.");
      return;
    }
    setRegionError(null);
    setRegions((current) => [...current, { ra_deg, dec_deg, radius_arcsec }]);
    setRegionRa("");
    setRegionDec("");
  }

  function removeRegion(index: number) {
    setRegions((current) => current.filter((_, i) => i !== index));
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
        queryRegions: regions,
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
        queryRegions: regions,
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
    if (activeProject.status !== "archived" && !window.confirm(`Archive “${activeProject.name}”? It will become read-only.`)) return;
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
    <Panel
      icon={FolderKanban}
      title="Projects"
      description={`${status}. Projects keep manifests, candidate reviews, experiments, results, and reports separate.`}
      actions={
        <Button onClick={() => void refresh(activeProject?.project_id)} disabled={busy} icon={RefreshCw} ariaLabel="Refresh projects">
          <span className="sr-only">Refresh</span>
        </Button>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(13rem,16rem)_1fr]">
        <div className="space-y-1 rounded border border-[var(--color-edge)] p-2">
          {projects.length === 0 && <p className="p-2 text-xs text-[var(--color-muted)]">No projects yet.</p>}
          {projects.map((project) => (
            <button
              type="button"
              key={project.project_id}
              onClick={() => onSelect(project)}
              aria-current={activeProject?.project_id === project.project_id ? "true" : undefined}
              className={`w-full rounded px-2 py-2 text-left text-xs ${activeProject?.project_id === project.project_id ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]" : "hover:bg-[var(--color-panel-2)]"}`}
            >
              <span className="block font-medium">{project.name}</span>
            <span className="mt-0.5 block truncate font-mono text-xs text-[var(--color-muted)]">{project.project_id} · {project.status}</span>
            </button>
          ))}
          <button type="button" onClick={() => { onSelect(null); setName(""); setDescription(""); setTags(""); setSelectedSurveys([]); setRegions([]); }} className="mt-2 flex w-full items-center justify-center gap-1 rounded border border-dashed border-[var(--color-edge)] px-2 py-1.5 text-xs text-[var(--color-muted)]">
            <Plus size={12} /> New project
          </button>
        </div>

        <div className="space-y-3">
          <Field id="project-name" label="Name" value={name} onChange={setName} width="w-full" placeholder="e.g. RR Lyrae follow-up…" />
          <label className="block text-xs text-[var(--color-muted)]">
            Description
            <textarea id="project-description" name="description" value={description} onChange={(event) => setDescription(event.target.value)} rows={3} maxLength={2000} className="mt-1 min-h-9 w-full resize-y rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2.5 py-1.5 text-sm text-[var(--color-text)] outline-none transition-colors focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40" />
          </label>
          <Field id="project-tags" label="Tags (comma-separated)" value={tags} onChange={setTags} width="w-full" placeholder="variable-stars, follow-up…" />
          <div>
            <p className="text-xs text-[var(--color-muted)]">Preferred surveys</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {surveys.map((survey) => {
                const key = survey.name.toLowerCase();
                const selected = selectedSurveys.includes(key);
                return <button type="button" key={key} onClick={() => toggleSurvey(key)} aria-pressed={selected} className={`min-h-9 rounded-full border px-3 py-1 text-xs ${selected ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]" : "border-[var(--color-edge)] text-[var(--color-muted)]"}`}>{survey.name}</button>;
              })}
            </div>
          </div>
          <div>
            <p className="text-xs text-[var(--color-muted)]">
              Sky regions ({regions.length}) — the pointings acquisition covers for this project
            </p>
            {regions.length > 0 && (
              <ul className="mt-1 space-y-1">
                {regions.map((region, index) => (
                  <li
                    key={`${region.ra_deg}-${region.dec_deg}-${index}`}
                    className="flex items-center justify-between rounded border border-[var(--color-edge)] px-2 py-1 font-mono text-[11px] text-[var(--color-muted)]"
                  >
                    <span>
                      {region.ra_deg.toFixed(4)}°, {region.dec_deg.toFixed(4)}° · r={region.radius_arcsec}″
                    </span>
                    <button
                      type="button"
                      onClick={() => removeRegion(index)}
                      className="rounded p-0.5 text-[var(--color-bad)] hover:bg-[var(--color-bad)]/10"
                      aria-label={`Remove region ${index + 1}`}
                    >
                      <X size={12} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="mt-1.5 grid gap-1.5 sm:grid-cols-[1fr_1fr_1fr_auto]">
              <Field id="region-ra" label="RA (deg)" value={regionRa} onChange={setRegionRa} width="w-full" min={0} max={359.999999} placeholder="RA (deg)" />
              <Field id="region-dec" label="Dec (deg)" value={regionDec} onChange={setRegionDec} width="w-full" min={-90} max={90} placeholder="Dec (deg)" />
              <Field id="region-radius" label="Radius (arcsec)" value={regionRadius} onChange={setRegionRadius} width="w-full" min={0.001} placeholder="Radius (arcsec)" />
              <Button onClick={addRegion} icon={Plus} tone="accent" className="self-end">
                Add
              </Button>
            </div>
            {regionError && <div className="mt-1"><Note tone="bad">{regionError}</Note></div>}
          </div>
          <div className="flex flex-wrap gap-2">
            {!activeProject && (
              <Button onClick={() => void create()} disabled={busy || !name.trim()} icon={Plus} tone="accent">
                Create project
              </Button>
            )}
            {activeProject && (
              <>
                <Button onClick={() => void save()} disabled={busy || activeProject.status === "archived" || !name.trim()} icon={Save} tone="accent">
                  Save
                </Button>
                <Button onClick={() => void archive()} disabled={busy} icon={Archive}>
                  {activeProject.status === "archived" ? "Restore" : "Archive"}
                </Button>
                <Button onClick={() => void validate()} disabled={busy} icon={CheckCircle2}>
                  Validate
                </Button>
              </>
            )}
          </div>
          {validation && (
            <Note tone={validation.valid ? "ok" : "bad"}>
              {validation.valid ? "Project layout and manifests are valid." : validation.issues.join(" ")}
            </Note>
          )}
          {activeProject && <p className="break-all font-mono text-[10px] text-[var(--color-muted)]">{activeProject.data_root}</p>}
        </div>
      </div>
    </Panel>
  );
}

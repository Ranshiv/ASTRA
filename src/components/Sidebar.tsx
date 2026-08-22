import {
  Activity,
  Compass,
  FlaskConical,
  Folders,
  LayoutDashboard,
  Link2,
  ListChecks,
  Radar,
  ScrollText,
  Settings,
  FileText,
  Brain,
  Radio,
  PanelLeftClose,
  PanelLeftOpen,
  Copy,
  Lock,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";

export type SectionId =
  | "dashboard"
  | "projects"
  | "acquire"
  | "events"
  | "sky"
  | "curves"
  | "candidates"
  | "crosssurvey"
  | "explain"
  | "experiments"
  | "models"
  | "reports"
  | "settings";

type NavItem = { id: SectionId; label: string; icon: LucideIcon };

/** Grouped by what a researcher is doing, which is the order plan section 41
 *  walks through: set up and collect, then look, then measure, then report. */
const NAV_GROUPS: { heading: string; items: NavItem[] }[] = [
  {
    heading: "Workflow",
    items: [
      { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
      { id: "projects", label: "Projects", icon: Folders },
      { id: "acquire", label: "Acquire", icon: Radar },
      { id: "events", label: "Events", icon: Radio },
    ],
  },
  {
    heading: "Investigate",
    items: [
      { id: "sky", label: "Sky explorer", icon: Compass },
      { id: "curves", label: "Light curves", icon: Activity },
      { id: "candidates", label: "Candidates", icon: ListChecks },
      { id: "crosssurvey", label: "Cross-survey", icon: Link2 },
      { id: "explain", label: "Explain", icon: ScrollText },
    ],
  },
  {
    heading: "Research",
    items: [
      { id: "experiments", label: "Experiments", icon: FlaskConical },
      { id: "models", label: "Models", icon: Brain },
      { id: "reports", label: "Reports", icon: FileText },
    ],
  },
  {
    heading: "System",
    items: [{ id: "settings", label: "Settings", icon: Settings }],
  },
];

const NARROW_QUERY = "(max-width: 1100px)";

export function Sidebar({
  active,
  onSelect,
  dataRoot,
  gatedSections,
  hasProject = true,
}: {
  active: SectionId;
  onSelect: (id: SectionId) => void;
  dataRoot?: string;
  gatedSections?: ReadonlySet<SectionId>;
  hasProject?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW_QUERY).matches);

  useEffect(() => {
    const query = window.matchMedia(NARROW_QUERY);
    const onChange = () => setNarrow(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  const effectiveCollapsed = collapsed || narrow;

  async function copyDataRoot() {
    if (!dataRoot) return;
    try {
      await navigator.clipboard?.writeText(dataRoot);
    } catch {
      // Clipboard access is optional in a desktop webview; the title still
      // exposes the complete path when copying is unavailable.
    }
  }

  return (
    <nav
      aria-label="Primary navigation"
      className={`${effectiveCollapsed ? "w-14" : "w-56"} flex shrink-0 flex-col overflow-y-auto border-r border-[var(--color-edge)] bg-[var(--color-void)] transition-[width] duration-200`}
    >
      <div className="flex items-center justify-end px-2 py-2">
        <button
          type="button"
          aria-label={effectiveCollapsed ? "Expand navigation" : "Collapse navigation"}
          aria-expanded={!effectiveCollapsed}
          disabled={narrow}
          onClick={() => setCollapsed((value) => !value)}
          className="flex min-h-9 min-w-9 items-center justify-center rounded text-[var(--color-muted)] hover:bg-[var(--color-panel)] hover:text-[var(--color-text)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {effectiveCollapsed ? <PanelLeftOpen aria-hidden="true" size={16} /> : <PanelLeftClose aria-hidden="true" size={16} />}
        </button>
      </div>
      <div className="flex-1 px-2 pb-4">
        {NAV_GROUPS.map((group) => (
          <div key={group.heading} className="mb-3">
            <p className={`px-2 pb-1.5 text-[11px] font-medium uppercase tracking-wider text-[var(--color-muted)] ${effectiveCollapsed ? "sr-only" : ""}`}>
              {group.heading}
            </p>
            <ul className="flex flex-col gap-0.5">
              {group.items.map(({ id, label, icon: Icon }) => {
                const isActive = id === active;
                const gated = Boolean(gatedSections?.has(id)) && !hasProject;
                return (
                  <li key={id}>
                    <button
                      type="button"
                      onClick={() => !gated && onSelect(id)}
                      disabled={gated}
                      aria-current={isActive ? "page" : undefined}
                      title={gated ? `${label} — requires an active project` : label}
                      className={`flex min-h-9 w-full items-center gap-2.5 rounded-md border-l-2 px-2.5 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-45 ${
                        isActive
                          ? "border-[var(--color-accent)] bg-[var(--color-panel)] text-[var(--color-text)]"
                          : "border-transparent text-[var(--color-muted)] hover:bg-[var(--color-panel)] hover:text-[var(--color-text)]"
                      }`}
                    >
                      <Icon
                        aria-hidden="true"
                        size={16}
                        strokeWidth={2}
                        className={isActive ? "text-[var(--color-accent)]" : ""}
                      />
                      <span className={`min-w-0 flex-1 truncate ${effectiveCollapsed ? "hidden" : "inline"}`}>{label}</span>
                      {gated && !effectiveCollapsed && (
                        <Lock aria-hidden="true" size={12} className="shrink-0 text-[var(--color-muted)]" />
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      {dataRoot && (
        <div className={`${effectiveCollapsed ? "hidden" : ""} border-t border-[var(--color-edge)] px-3 py-3`}>
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] uppercase tracking-wider text-[var(--color-muted)]">Data root</p>
            <button
              type="button"
              onClick={() => void copyDataRoot()}
              title="Copy data root"
              aria-label="Copy data root"
              className="rounded p-1 text-[var(--color-muted)] hover:bg-[var(--color-panel)] hover:text-[var(--color-text)]"
            >
              <Copy size={12} />
            </button>
          </div>
          <p className="mt-1 truncate font-mono text-[11px] text-[var(--color-muted)]" title={dataRoot}>
            {dataRoot}
          </p>
        </div>
      )}
    </nav>
  );
}

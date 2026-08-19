import {
  Activity,
  Compass,
  FlaskConical,
  Folders,
  LayoutDashboard,
  Link2,
  ListChecks,
  Radar,
  Settings,
  FileText,
  Brain,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type SectionId =
  | "dashboard"
  | "projects"
  | "acquire"
  | "sky"
  | "curves"
  | "candidates"
  | "crosssurvey"
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
    ],
  },
  {
    heading: "Investigate",
    items: [
      { id: "sky", label: "Sky explorer", icon: Compass },
      { id: "curves", label: "Light curves", icon: Activity },
      { id: "candidates", label: "Candidates", icon: ListChecks },
      { id: "crosssurvey", label: "Cross-survey", icon: Link2 },
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

export function Sidebar({
  active,
  onSelect,
  dataRoot,
}: {
  active: SectionId;
  onSelect: (id: SectionId) => void;
  dataRoot?: string;
}) {
  return (
    <nav className="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-[var(--color-edge)] bg-[var(--color-void)]">
      <div className="flex-1 px-2 py-4">
        {NAV_GROUPS.map((group) => (
          <div key={group.heading} className="mb-3">
            <p className="px-2 pb-1.5 text-[11px] font-medium uppercase tracking-wider text-[var(--color-muted)]">
              {group.heading}
            </p>
            <ul className="flex flex-col gap-0.5">
              {group.items.map(({ id, label, icon: Icon }) => {
                const isActive = id === active;
                return (
                  <li key={id}>
                    <button
                      type="button"
                      onClick={() => onSelect(id)}
                      aria-current={isActive ? "page" : undefined}
                      className={`flex w-full items-center gap-2.5 rounded-md border-l-2 px-2.5 py-2 text-sm transition-colors ${
                        isActive
                          ? "border-[var(--color-accent)] bg-[var(--color-panel)] text-[var(--color-text)]"
                          : "border-transparent text-[var(--color-muted)] hover:bg-[var(--color-panel)] hover:text-[var(--color-text)]"
                      }`}
                    >
                      <Icon
                        size={16}
                        strokeWidth={2}
                        className={isActive ? "text-[var(--color-accent)]" : ""}
                      />
                      {label}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      {dataRoot && (
        <div className="border-t border-[var(--color-edge)] px-3 py-3">
          <p className="text-[10px] uppercase tracking-wider text-[var(--color-muted)]">Data root</p>
          <p className="mt-1 truncate font-mono text-[11px] text-[var(--color-muted)]" title={dataRoot}>
            {dataRoot}
          </p>
        </div>
      )}
    </nav>
  );
}

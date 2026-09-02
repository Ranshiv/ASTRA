import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function SectionHeader({
  icon: Icon,
  title,
  description,
  actions,
}: {
  icon: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-[var(--color-edge)] bg-[var(--color-panel-2)] text-[var(--color-accent)]">
          <Icon size={16} strokeWidth={2} />
        </span>
        <div>
          <h2 className="text-base font-semibold tracking-tight">{title}</h2>
          {description && (
            <p className="mt-0.5 text-sm text-[var(--color-muted)]">{description}</p>
          )}
        </div>
      </div>
      {/* items-end, not items-center: actions mix single-row Buttons with
          two-row Field/Select (label above input) -- centering by total
          height puts a Button's middle at the gap between label and input
          instead of lining it up with the input box. Bottom-aligning keeps
          every same-height group identical (no visible change there) and
          fixes the mixed-height case. */}
      {actions && <div className="flex flex-wrap items-end gap-1.5">{actions}</div>}
    </div>
  );
}

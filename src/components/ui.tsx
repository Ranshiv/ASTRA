/** Shared primitives for the plan section 10 views.
 *
 * Extracted rather than repeated: seven views appeared at once, and without a
 * common Panel/Button/Field the same Tailwind class strings would have been
 * pasted seven times and drifted apart on the first restyle.
 */
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";

import { SectionHeader } from "@/components/SectionHeader";
import {
  Select as SelectPrimitive,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function Panel({
  icon,
  title,
  description,
  actions,
  children,
}: {
  icon: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
      <SectionHeader icon={icon} title={title} description={description} actions={actions} />
      {children && <div className="mt-3">{children}</div>}
    </section>
  );
}

const BUTTON_SIZES = {
  sm: "min-h-8 px-2.5 py-1",
  md: "min-h-9 px-3 py-1.5",
} as const;

export function Button({
  children,
  onClick,
  disabled,
  icon: Icon,
  tone = "default",
  size = "md",
  title,
  ariaLabel,
  loading = false,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  icon?: LucideIcon;
  tone?: "default" | "accent";
  size?: "sm" | "md";
  title?: string;
  ariaLabel?: string;
  loading?: boolean;
  className?: string;
}) {
  const border =
    tone === "accent"
      ? "border-[var(--color-accent)] text-[var(--color-accent)]"
      : "border-[var(--color-edge)] text-[var(--color-text)]";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      title={title}
      aria-label={ariaLabel}
      aria-busy={loading || undefined}
      className={`inline-flex items-center justify-center gap-1.5 rounded border text-xs transition-colors hover:border-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)] disabled:cursor-not-allowed disabled:opacity-40 ${BUTTON_SIZES[size]} ${border} ${className}`}
    >
      {Icon && <Icon aria-hidden="true" size={12} strokeWidth={2} />}
      {loading ? "Working…" : children}
    </button>
  );
}

export function Field({
  label,
  value,
  onChange,
  placeholder,
  width = "w-28",
  type = "text",
  name,
  id,
  help,
  error,
  required = false,
  min,
  max,
  step,
  autoComplete,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  width?: string;
  type?: string;
  name?: string;
  id?: string;
  help?: ReactNode;
  error?: ReactNode;
  required?: boolean;
  min?: number | string;
  max?: number | string;
  step?: number | string;
  autoComplete?: string;
}) {
  const generatedId = useId();
  const inputId = id ?? `field-${generatedId}`;
  const helpId = help ? `${inputId}-help` : undefined;
  const errorId = error ? `${inputId}-error` : undefined;
  return (
    <label htmlFor={inputId} className="flex min-w-0 flex-col gap-1 text-xs text-[var(--color-muted)]">
      <span>{label}{required && <span aria-hidden="true"> · required</span>}</span>
      <input
        id={inputId}
        name={name}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        required={required}
        min={min}
        max={max}
        step={step}
        autoComplete={autoComplete}
        aria-invalid={Boolean(error)}
        aria-describedby={[helpId, errorId].filter(Boolean).join(" ") || undefined}
        className={`${width} min-h-9 rounded border ${error ? "border-[var(--color-bad)]" : "border-[var(--color-edge)]"} bg-[var(--color-void)] px-2.5 py-1.5 text-sm text-[var(--color-text)] outline-none transition-colors focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40`}
      />
      {help && <span id={helpId} className="text-[var(--color-muted)]">{help}</span>}
      {error && <span id={errorId} className="text-[var(--color-bad)]" role="alert">{error}</span>}
    </label>
  );
}

// Radix's Select.Item reserves the empty string internally to mean "no
// selection" -- an option genuinely valued "" (e.g. CrossSurveyPanel's
// "Automatic · largest catalogue" default) is silently unmatchable, so
// SelectValue renders blank forever, even though the right option is
// selected. Map "" to this sentinel only at the Radix boundary, so every
// caller of Select can keep passing/receiving "" exactly as before.
const EMPTY_SELECT_VALUE = "__select-empty__";

export function Select({
  label,
  value,
  options,
  onChange,
  id,
  name,
  help,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  id?: string;
  name?: string;
  help?: ReactNode;
}) {
  const generatedId = useId();
  const selectId = id ?? `select-${generatedId}`;
  const helpId = help ? `${selectId}-help` : undefined;
  return (
    // A styled Radix listbox, not a native <select> -- a native select's
    // open options popup can't be restyled via CSS in Tauri's WebView2,
    // which is what made every dropdown in the app look like unstyled OS
    // chrome once opened. The external API (props) is unchanged, so every
    // existing caller of Select needed no changes.
    <div className="flex min-w-0 flex-col gap-1 text-xs text-[var(--color-muted)]">
      <span id={`${selectId}-label`}>{label}</span>
      <SelectPrimitive
        value={value === "" ? EMPTY_SELECT_VALUE : value}
        onValueChange={(next) => onChange(next === EMPTY_SELECT_VALUE ? "" : next)}
        name={name}
      >
        <SelectTrigger
          id={selectId}
          aria-labelledby={`${selectId}-label`}
          aria-describedby={helpId}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem
              key={option.value}
              value={option.value === "" ? EMPTY_SELECT_VALUE : option.value}
            >
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </SelectPrimitive>
      {help && <span id={helpId}>{help}</span>}
    </div>
  );
}

export function KeyValue({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <dl className="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-1 text-xs">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-[var(--color-muted)]">{key}</dt>
          <dd className="break-all font-mono text-[var(--color-text)]">{value ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Empty({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded border border-dashed border-[var(--color-edge)] bg-[var(--color-void)]/40 p-4 text-sm text-[var(--color-muted)]">
      <p>{children}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function Note({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" | "bad" | "ok" }) {
  const colour = {
    muted: "text-[var(--color-muted)]",
    warn: "text-[var(--color-warn)]",
    bad: "text-[var(--color-bad)]",
    ok: "text-[var(--color-ok)]",
  }[tone];
  return <p className={`text-sm ${colour}`} role={tone === "bad" ? "alert" : "status"}>{children}</p>;
}

export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" | "bad" | "ok" | "accent" }) {
  const colour = {
    muted: "border-[var(--color-edge)] text-[var(--color-muted)]",
    warn: "border-[var(--color-warn)]/50 text-[var(--color-warn)]",
    bad: "border-[var(--color-bad)]/50 text-[var(--color-bad)]",
    ok: "border-[var(--color-ok)]/50 text-[var(--color-ok)]",
    accent: "border-[var(--color-accent)]/50 text-[var(--color-accent)]",
  }[tone];
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide ${colour}`}>
      {children}
    </span>
  );
}

export function Table({ head, children, caption }: { head: string[]; children: ReactNode; caption?: string }) {
  return (
    <div className="overflow-x-auto rounded border border-[var(--color-edge)]">
      <table className="w-full min-w-[32rem] text-left text-sm">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-[var(--color-edge)] text-[var(--color-muted)]">
            {head.map((cell) => (
              <th key={cell} scope="col" className="sticky top-0 bg-[var(--color-panel-2)] px-2.5 py-2 font-medium">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** A number that may legitimately be absent. Renders an em dash, never 0. */
export function num(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toFixed(digits);
}

/** Load once on mount, tracking the three states a call actually has.
 *
 * `error` is kept distinct from `data === null`: an engine that has not been
 * asked yet and an engine that refused are different situations, and the views
 * say which one they are in rather than showing an empty table for both.
 */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const mounted = useRef(true);

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const result = await load();
      if (mounted.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      if (mounted.current) setError(String(err));
    } finally {
      if (mounted.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mounted.current = true;
    void run();
    return () => {
      mounted.current = false;
    };
  }, [run]);

  return { data, error, loading, reload: run };
}

/** Drive one long-running engine action, keeping its status message. */
export function useAction(initial = "") {
  const [status, setStatus] = useState(initial);
  const [busy, setBusy] = useState(false);

  const run = useCallback(
    async (message: string, work: () => Promise<string | void>) => {
      setBusy(true);
      setStatus(message);
      try {
        setStatus((await work()) || "Done.");
      } catch (err) {
        setStatus(String(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  return { status, busy, run, setStatus };
}

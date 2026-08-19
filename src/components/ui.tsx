/** Shared primitives for the plan section 10 views.
 *
 * Extracted rather than repeated: seven views appeared at once, and without a
 * common Panel/Button/Field the same Tailwind class strings would have been
 * pasted seven times and drifted apart on the first restyle.
 */
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { SectionHeader } from "@/components/SectionHeader";

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

export function Button({
  children,
  onClick,
  disabled,
  icon: Icon,
  tone = "default",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  icon?: LucideIcon;
  tone?: "default" | "accent";
  title?: string;
}) {
  const border =
    tone === "accent"
      ? "border-[var(--color-accent)] text-[var(--color-accent)]"
      : "border-[var(--color-edge)] text-[var(--color-text)]";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs transition hover:border-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-40 ${border}`}
    >
      {Icon && <Icon size={12} strokeWidth={2} />}
      {children}
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
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  width?: string;
  type?: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-[var(--color-muted)]">
      {label}
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className={`${width} rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1 text-xs text-[var(--color-text)]`}
      />
    </label>
  );
}

export function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-[var(--color-muted)]">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-1 text-xs text-[var(--color-text)]"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
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

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-xs text-[var(--color-muted)]">{children}</p>;
}

export function Note({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" | "bad" | "ok" }) {
  const colour = {
    muted: "text-[var(--color-muted)]",
    warn: "text-[var(--color-warn)]",
    bad: "text-[var(--color-bad)]",
    ok: "text-[var(--color-ok)]",
  }[tone];
  return <p className={`text-xs ${colour}`}>{children}</p>;
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

export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[32rem] text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--color-edge)] text-[var(--color-muted)]">
            {head.map((cell) => (
              <th key={cell} className="px-2 py-1.5 font-medium">
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

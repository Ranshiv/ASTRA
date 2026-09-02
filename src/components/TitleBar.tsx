import { Copy, Minus, Monitor, Moon, Square, Sun, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

import { useTheme, type ThemeChoice } from "@/lib/theme";

const tauriInternals = typeof window !== "undefined"
  ? (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__
  : undefined;
const appWindow = tauriInternals ? getCurrentWindow() : null;

function TitleBarButton({
  label,
  onClick,
  danger,
  disabled = false,
  children,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
      className={`flex h-full w-11 items-center justify-center text-[var(--color-muted)] transition-colors hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-45 ${
        danger ? "hover:bg-[var(--color-bad)] hover:text-white" : "hover:bg-[var(--color-edge)]"
      }`}
    >
      {children}
    </button>
  );
}

const THEME_ICON: Record<ThemeChoice, typeof Monitor> = {
  system: Monitor,
  light: Sun,
  dark: Moon,
};

const THEME_NEXT_LABEL: Record<ThemeChoice, string> = {
  system: "light",
  light: "dark",
  dark: "system",
};

function ThemeToggle() {
  const { choice, resolved, cycle } = useTheme();
  const Icon = THEME_ICON[choice];

  // Follows the resolved theme into WebView2's native chrome (scrollbars,
  // context menus) -- purely cosmetic outside Tauri, so it's guarded the
  // same way TitleBar's window controls are (appWindow is null in the
  // browser dev path). Requires core:window:allow-set-theme, granted in
  // src-tauri/capabilities/default.json.
  useEffect(() => {
    void appWindow?.setTheme(resolved).catch(() => {
      /* native chrome theming is a cosmetic extra; the CSS theme still applies */
    });
  }, [resolved]);

  return (
    <TitleBarButton
      label={`Theme: ${choice}. Switch to ${THEME_NEXT_LABEL[choice]}.`}
      onClick={cycle}
    >
      <Icon size={14} />
    </TitleBarButton>
  );
}

export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    if (!appWindow) return;

    // `onResized` resolves asynchronously; if the component unmounts before
    // it settles, `cancelled` makes the resolved handler tear itself down
    // immediately instead of assigning `unlisten` after cleanup already ran
    // (which would otherwise leak the listener for the app's lifetime).
    let cancelled = false;
    let unlisten: (() => void) | undefined;

    appWindow.isMaximized().then(setMaximized);
    appWindow.onResized(() => {
      appWindow.isMaximized().then(setMaximized);
    }).then((fn) => {
      if (cancelled) {
        fn();
        return;
      }
      unlisten = fn;
    });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // F11 is the OS-wide convention for toggling fullscreen; wired here at the
  // app root (TitleBar mounts once) rather than per-view so it works no
  // matter what's focused. Requires core:window:allow-set-fullscreen /
  // allow-is-fullscreen, granted in src-tauri/capabilities/default.json.
  useEffect(() => {
    if (!appWindow) return;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "F11") return;
      event.preventDefault();
      void appWindow!.isFullscreen().then((current) => appWindow!.setFullscreen(!current));
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className="flex h-9 shrink-0 select-none items-center justify-between border-b border-[var(--color-edge)] bg-[var(--color-void)]">
      <div data-tauri-drag-region className="flex h-full flex-1 items-center gap-2 pl-3">
        <img
          src="/astra_icon.svg"
          alt=""
          aria-hidden="true"
          width={18}
          height={18}
          draggable={false}
          className="shrink-0 rounded-[5px]"
        />
        <span className="text-xs font-medium tracking-wide text-[var(--color-muted)]">ASTRA</span>
      </div>

      <div className="flex h-full items-center">
        <ThemeToggle />
        <TitleBarButton disabled={!appWindow} label="Minimize" onClick={() => void appWindow?.minimize()}>
          <Minus size={14} />
        </TitleBarButton>
        <TitleBarButton
          label={maximized ? "Restore" : "Maximize"}
          disabled={!appWindow}
          onClick={() => void appWindow?.toggleMaximize()}
        >
          {maximized ? <Copy size={13} /> : <Square size={12} />}
        </TitleBarButton>
        <TitleBarButton disabled={!appWindow} label="Close" onClick={() => void appWindow?.close()} danger>
          <X size={15} />
        </TitleBarButton>
      </div>
    </div>
  );
}

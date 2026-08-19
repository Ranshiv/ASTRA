import { Copy, Minus, Radar, Square, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const appWindow = getCurrentWindow();

function TitleBarButton({
  label,
  onClick,
  danger,
  children,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className={`flex h-full w-11 items-center justify-center text-[var(--color-muted)] transition-colors hover:text-[var(--color-text)] ${
        danger ? "hover:bg-[var(--color-bad)] hover:text-white" : "hover:bg-[var(--color-edge)]"
      }`}
    >
      {children}
    </button>
  );
}

export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    appWindow.isMaximized().then(setMaximized);
    appWindow.onResized(() => {
      appWindow.isMaximized().then(setMaximized);
    }).then((fn) => {
      unlisten = fn;
    });

    return () => unlisten?.();
  }, []);

  return (
    <div className="flex h-9 shrink-0 select-none items-center justify-between border-b border-[var(--color-edge)] bg-[var(--color-void)]">
      <div data-tauri-drag-region className="flex h-full flex-1 items-center gap-2 pl-3">
        <Radar size={13} strokeWidth={2} className="text-[var(--color-accent)]" />
        <span className="text-xs font-medium tracking-wide text-[var(--color-muted)]">ASTRA</span>
      </div>

      <div className="flex h-full items-center">
        <TitleBarButton label="Minimize" onClick={() => appWindow.minimize()}>
          <Minus size={14} />
        </TitleBarButton>
        <TitleBarButton
          label={maximized ? "Restore" : "Maximize"}
          onClick={() => appWindow.toggleMaximize()}
        >
          {maximized ? <Copy size={13} /> : <Square size={12} />}
        </TitleBarButton>
        <TitleBarButton label="Close" onClick={() => appWindow.close()} danger>
          <X size={15} />
        </TitleBarButton>
      </div>
    </div>
  );
}

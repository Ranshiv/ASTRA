import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * A failed initial render used to leave the native window indistinguishable
 * from an uninitialised WebView. Keep the recovery UI dependency-free so it
 * can render even when a lazily loaded workspace module fails.
 */
export class StartupErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ASTRA failed to render", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="mx-auto flex min-h-full max-w-3xl flex-col gap-3 px-8 py-10">
          <h1 className="text-3xl font-semibold tracking-tight">ASTRA could not start</h1>
          <p className="text-sm text-[var(--color-muted)]">
            The desktop shell loaded, but the application UI encountered an error.
          </p>
          <pre className="overflow-auto rounded-lg border border-[var(--color-bad)] bg-[var(--color-panel)] p-4 font-mono text-xs text-[var(--color-bad)]">
            {this.state.error.message || "Unknown startup error"}
          </pre>
          <p className="text-xs text-[var(--color-muted)]">
            Restart ASTRA. If this screen returns, include the message above with the crash report.
          </p>
        </main>
      );
    }

    return this.props.children;
  }
}

import { getCurrentWindow } from "@tauri-apps/api/window";

// Deliberately its own module, loaded by its own <script> tag ahead of
// main.tsx in index.html. ES modules resolve their entire import graph
// before running any of their own top-level code, so a `show()` call placed
// inside main.tsx -- however early in the file -- cannot run until React,
// ReactDOM, and the whole App component tree have finished loading. In dev
// mode, where Vite serves that graph unbundled, that waterfall is exactly
// the multi-second gap this file exists to skip: by the time this executes,
// index.html's static boot-log markup has already painted, so revealing the
// window here shows real content immediately instead of waiting on the rest
// of the app.
// Vite's browser preview does not provide Tauri's internals object. Guard the
// native-only reveal call so the regular web page can still render for UI
// development and visual checks.
const tauriInternals = typeof window !== "undefined"
  ? (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__
  : undefined;

if (tauriInternals) {
  getCurrentWindow()
    .show()
    .catch((error) => console.error("failed to show the main window", error));
}

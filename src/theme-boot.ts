import { applyStoredTheme } from "@/lib/theme";

// Loaded as the very first module script in index.html, ahead of
// reveal-window.ts and main.tsx. Module scripts run in document order, and
// reveal-window.ts is what calls getCurrentWindow().show() on a window
// configured "visible": false -- so stamping `data-theme` here happens
// strictly before the Tauri window is ever shown, and before React mounts in
// the browser dev path. That is what avoids a themed flash on launch.
//
// This can't be an inline <script> in <head> instead: tauri.conf.json's CSP
// is `script-src 'self' 'wasm-unsafe-eval'` with no `'unsafe-inline'`, so an
// inline script is blocked in the desktop window.
applyStoredTheme();

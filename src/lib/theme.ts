import { useCallback, useSyncExternalStore } from "react";

/** Theme state: read/write localStorage, resolve "system" against
 * `prefers-color-scheme`, and stamp the resolved value onto `<html
 * data-theme>` -- the attribute src/index.css keys its `[data-theme="light"]`
 * override off of.
 *
 * Modelled on src/lib/navigation.ts's useHashNavigation: state that lives
 * outside React (here, localStorage + the DOM attribute, there, the URL
 * hash), read with a lazy initializer, written through one function, and
 * kept in sync with external changes (a matchMedia listener here, popstate/
 * hashchange there) via a subscribed effect. Unlike navigation, theme has
 * more than one consumer (TitleBar's toggle, and in future a Settings panel)
 * on separate branches of the component tree, so it is exposed as a small
 * module-level store consumed via useSyncExternalStore rather than drilled
 * through props.
 */

export type ThemeChoice = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "astra.theme";
const MEDIA_QUERY = "(prefers-color-scheme: dark)";

function safeLocalStorage(): Storage | null {
  // Some restricted webviews (and privacy modes) throw on access, not just
  // on read/write -- so the whole thing is guarded, not just getItem/setItem.
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function readStoredChoice(): ThemeChoice {
  const value = safeLocalStorage()?.getItem(STORAGE_KEY);
  return value === "light" || value === "dark" || value === "system" ? value : "system";
}

function prefersDark(): boolean {
  try {
    return window.matchMedia(MEDIA_QUERY).matches;
  } catch {
    return false;
  }
}

function resolve(choice: ThemeChoice): ResolvedTheme {
  return choice === "system" ? (prefersDark() ? "dark" : "light") : choice;
}

let choice: ThemeChoice = "system";
let resolved: ResolvedTheme = "light";
const listeners = new Set<() => void>();

function notify() {
  for (const listener of listeners) listener();
}

function applyToDocument() {
  document.documentElement.dataset.theme = resolved;
}

/** Set the initial theme before first paint. Called from theme-boot.ts,
 * loaded ahead of reveal-window.ts and main.tsx in index.html so the
 * `data-theme` attribute lands before the (initially hidden) Tauri window is
 * ever shown, and before React mounts in the browser dev path. Safe to call
 * more than once -- useTheme() below calls it again lazily so a component
 * that mounts without theme-boot.ts having run (e.g. in a unit test) still
 * initializes correctly. */
export function applyStoredTheme(): void {
  choice = readStoredChoice();
  resolved = resolve(choice);
  applyToDocument();
}

export function setThemeChoice(next: ThemeChoice): void {
  choice = next;
  resolved = resolve(choice);
  try {
    safeLocalStorage()?.setItem(STORAGE_KEY, choice);
  } catch {
    /* persistence is a convenience; the in-memory choice still applies */
  }
  applyToDocument();
  notify();
}

const CYCLE: Record<ThemeChoice, ThemeChoice> = {
  system: "light",
  light: "dark",
  dark: "system",
};

export function cycleThemeChoice(): void {
  setThemeChoice(CYCLE[choice]);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Only one OS-level listener is needed regardless of how many components
  // subscribe: it just re-resolves "system" and notifies everyone.
  let media: MediaQueryList | null = null;
  const onMediaChange = () => {
    if (choice !== "system") return;
    resolved = resolve(choice);
    applyToDocument();
    notify();
  };
  try {
    media = window.matchMedia(MEDIA_QUERY);
    media.addEventListener("change", onMediaChange);
  } catch {
    media = null;
  }
  return () => {
    listeners.delete(listener);
    media?.removeEventListener("change", onMediaChange);
  };
}

function getChoiceSnapshot(): ThemeChoice {
  return choice;
}

function getResolvedSnapshot(): ResolvedTheme {
  return resolved;
}

/** React hook for theme state. Initializes from storage on first use (a
 * no-op if theme-boot.ts already ran) so the value is correct even before
 * any component has rendered. */
export function useTheme(): {
  choice: ThemeChoice;
  resolved: ResolvedTheme;
  setChoice: (choice: ThemeChoice) => void;
  cycle: () => void;
} {
  if (typeof document !== "undefined" && !document.documentElement.dataset.theme) {
    applyStoredTheme();
  }
  const choiceValue = useSyncExternalStore(subscribe, getChoiceSnapshot);
  const resolvedValue = useSyncExternalStore(subscribe, getResolvedSnapshot);
  const setChoice = useCallback((next: ThemeChoice) => setThemeChoice(next), []);
  const cycle = useCallback(() => cycleThemeChoice(), []);
  return { choice: choiceValue, resolved: resolvedValue, setChoice, cycle };
}

/** Read a `--color-*` custom property's resolved value off `<html>`, as the
 * literal 6-digit hex string declared in src/index.css. Used to bridge the
 * CSS token system into contexts that cannot take a `var()`: Plotly configs
 * (LightCurvePlot), Three.js materials (SpatialScene3D), and Aladin Lite
 * init options (AladinSky). */
export function readThemeColor(name: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || "#000000";
}

/** Same lookup, as a 0xRRGGBB integer for three.js material colours. Falls
 * back to black on a malformed value rather than throwing. */
export function readThemeColorHexInt(name: string): number {
  const hex = readThemeColor(name).replace(/^#/, "");
  const parsed = parseInt(hex, 16);
  return Number.isFinite(parsed) ? parsed : 0x000000;
}

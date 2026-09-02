/** src/lib/theme.ts: choice persistence, system resolution, and the
 * data-theme attribute it stamps onto <html> for src/index.css's
 * `[data-theme="light"]` override to key off of. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { applyStoredTheme, cycleThemeChoice, setThemeChoice } from "@/lib/theme";

const STORAGE_KEY = "astra.theme";

function mockMatchMedia(matches: boolean) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mql = {
    matches,
    media: "(prefers-color-scheme: dark)",
    addEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => {
      listeners.add(listener);
    },
    removeEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => {
      listeners.delete(listener);
    },
  } as unknown as MediaQueryList;
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue(mql));
  return {
    fire: (next: boolean) => {
      (mql as { matches: boolean }).matches = next;
      listeners.forEach((listener) => listener({ matches: next } as MediaQueryListEvent));
    },
  };
}

describe("theme", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.theme;
    mockMatchMedia(false);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to system, resolved against prefers-color-scheme", () => {
    mockMatchMedia(true);
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("resolves system to light when the OS prefers light", () => {
    mockMatchMedia(false);
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("persists an explicit choice and stamps it on <html>", () => {
    setThemeChoice("light");
    expect(localStorage.getItem(STORAGE_KEY)).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");

    setThemeChoice("dark");
    expect(localStorage.getItem(STORAGE_KEY)).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("reads a stored choice back on the next applyStoredTheme() call", () => {
    setThemeChoice("dark");
    delete document.documentElement.dataset.theme;
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("ignores a corrupted storage value and falls back to system", () => {
    localStorage.setItem(STORAGE_KEY, "not-a-real-choice");
    mockMatchMedia(false);
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("cycles system -> light -> dark -> system", () => {
    setThemeChoice("system");
    cycleThemeChoice();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("light");
    cycleThemeChoice();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("dark");
    cycleThemeChoice();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("system");
  });
});

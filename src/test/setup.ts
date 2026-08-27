import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

/** jsdom implements neither of these, and Radix's Select (ui/select.tsx)
 * calls them on open/scroll. Without these stubs any test that opens a
 * Select throws inside jsdom rather than failing on real app behaviour. */
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
}
if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => {};
}
if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = () => {};
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

/** Every Tauri command goes through this one mock.
 *
 * Mocking at the `invoke` boundary rather than at `engine.*` is deliberate: it
 * leaves the wrappers in `src/lib/engine.ts` under test, so a wrapper that
 * sends the wrong argument name — which the Rust side would silently receive
 * as `None` — fails here instead of at runtime.
 */
export const invoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invoke(...args),
}));

afterEach(() => {
  cleanup();
  invoke.mockReset();
});

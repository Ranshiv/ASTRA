import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

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

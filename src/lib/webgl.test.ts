/** src/lib/webgl.ts: jsdom has no WebGL implementation at all --
 * HTMLCanvasElement.getContext('webgl'|'webgl2') returns null there -- so the
 * only thing worth asserting from a jsdom test is that the probe degrades to
 * false/false/null rather than throwing, which is exactly the failure mode
 * this module exists to replace (see LightCurvePlot.tsx's history: a bare
 * getContext() call that returned a false positive in a real webview). */
import { describe, expect, it } from "vitest";

import { webglSupport } from "@/lib/webgl";

describe("webglSupport", () => {
  it("reports no WebGL support without throwing in an environment with no implementation", () => {
    const result = webglSupport();
    expect(result).toEqual({ webgl1: false, webgl2: false, renderer: null });
  });

  it("is memoised: repeated calls return the same object", () => {
    expect(webglSupport()).toBe(webglSupport());
  });
});

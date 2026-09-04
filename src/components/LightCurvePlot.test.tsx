/** LightCurvePlot's WebGL-failure detection.
 *
 * Plotly signals a failed `scattergl`/regl init only by inserting its own
 * `div.no-webgl` placeholder into the plot container -- it neither throws nor
 * rejects. That silent-DOM-only failure is what put a grey "WebGL is not
 * supported" box over real data in packaged builds even on machines whose
 * WebGL works, so the detection is worth pinning down. */
import { describe, expect, it } from "vitest";

import { hasNoWebglPlaceholder } from "@/components/LightCurvePlot";

describe("hasNoWebglPlaceholder", () => {
  it("detects Plotly's placeholder so the caller can redraw via SVG", () => {
    const container = document.createElement("div");
    const placeholder = document.createElement("div");
    placeholder.className = "no-webgl";
    placeholder.appendChild(document.createElement("p"));
    container.appendChild(placeholder);

    expect(hasNoWebglPlaceholder(container)).toBe(true);
  });

  it("reports nothing for a container holding a real plot", () => {
    const container = document.createElement("div");
    const plot = document.createElement("div");
    plot.className = "plot-container plotly";
    container.appendChild(plot);

    expect(hasNoWebglPlaceholder(container)).toBe(false);
  });

  it("reports nothing for an empty container", () => {
    expect(hasNoWebglPlaceholder(document.createElement("div"))).toBe(false);
  });
});

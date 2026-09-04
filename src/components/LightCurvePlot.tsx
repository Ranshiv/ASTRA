import { useEffect, useRef, useState } from "react";
import type Plotly from "plotly.js-dist-min";

import type { CurvePayload, FoldedCurve } from "@/lib/engine";
import { readThemeColor, useTheme } from "@/lib/theme";
import { webglSupport } from "@/lib/webgl";

/** Band colours read from the theme tokens at render time rather than
 * hardcoded: they were already exact duplicates of --color-ok/-bad/-warn/
 * -accent/-text (g/r/i/TESS/G respectively), so sourcing them from
 * getComputedStyle makes them follow a theme change for free instead of
 * staying frozen at whatever the dark palette happened to be. */
function bandColors(): Record<string, string> {
  return {
    g: readThemeColor("--color-ok"),
    r: readThemeColor("--color-bad"),
    i: readThemeColor("--color-warn"),
    TESS: readThemeColor("--color-accent"),
    G: readThemeColor("--color-text"),
  };
}

/** Convert a #rrggbb colour into rgba() so error bars can be faded. Falls
 * back to opaque black on a malformed value instead of propagating NaN into
 * the rgba() string, which Plotly would silently fail to parse. */
function withAlpha(hex: string, alpha: number): string {
  const value = parseInt(hex.slice(1), 16);
  if (!Number.isFinite(value)) return `rgba(0,0,0,${alpha})`;
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function layoutBase(): Partial<Plotly.Layout> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: readThemeColor("--color-muted"), size: 11 },
    margin: { l: 56, r: 16, t: 8, b: 40 },
    showlegend: false,
    hovermode: "closest",
  };
}

/** Plotly reports a failed WebGL/regl init only by drawing its own
 * `div.no-webgl` placeholder into the plot container -- it neither throws nor
 * rejects, so no try/catch or `.catch()` can see it. Detecting that element is
 * the only reliable signal that `scattergl` silently gave up. */
export function hasNoWebglPlaceholder(node: HTMLElement): boolean {
  return !!node.querySelector(".no-webgl");
}

function axisBase(): Partial<Plotly.LayoutAxis> {
  const edge = readThemeColor("--color-edge");
  return {
    gridcolor: edge,
    zerolinecolor: edge,
    linecolor: edge,
  };
}

export function LightCurvePlot({
  curve,
  folded,
}: {
  curve: CurvePayload;
  folded: FoldedCurve | null;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const { resolved } = useTheme();

  useEffect(() => {
    if (!container.current) return;
    setRenderError(null);

    let disposed = false;
    const render = async () => {
      const { default: Plotly } = await import("plotly.js-dist-min");
      if (disposed || !container.current) return;
      const color = bandColors()[curve.band] ?? readThemeColor("--color-accent");
    const showing = folded ?? curve;
    const x = folded ? folded.phase : curve.time;
    const y = showing.value;

    const trace: Partial<Plotly.PlotData> = {
      x,
      y,
      // `scattergl` needs a real WebGL context (see src/lib/webgl.ts for why
      // a naive getContext() probe isn't enough). Where WebGL isn't usable,
      // Plotly doesn't throw or fall back on its own -- it draws a "WebGL is
      // not supported" placeholder *inside* the plot area, which the
      // render() try/catch below never sees because nothing actually failed.
      type: webglSupport().webgl1 ? "scattergl" : "scatter",
      mode: "markers",
      marker: { size: folded ? 3 : 4, color, opacity: 0.75 },
      hovertemplate: folded
        ? "phase %{x:.4f}<br>%{y:.4f}<extra></extra>"
        : "%{x:.6f}<br>%{y:.4f}<extra></extra>",
    };

    // Error bars only make sense on the unfolded series, where each point
    // still corresponds to one epoch. Transparency comes from the colour's
    // alpha channel; Plotly's error bars have no opacity field of their own.
    if (!folded && curve.value_err.length === curve.value.length) {
      trace.error_y = {
        type: "data",
        array: curve.value_err,
        visible: true,
        color: withAlpha(color, 0.35),
        thickness: 0.6,
        width: 0,
      };
    }

    const layout: Partial<Plotly.Layout> = {
      ...layoutBase(),
      xaxis: {
        ...axisBase(),
        title: {
          text: folded ? "Phase" : `Time (${curve.time_system})`,
          font: { size: 11 },
        },
      },
      yaxis: {
        ...axisBase(),
        title: {
          text: curve.value_kind === "mag" ? "Magnitude" : "Flux",
          font: { size: 11 },
        },
        // Magnitudes run backwards: brighter is a smaller number, so an
        // un-inverted axis would show every outburst as a dip.
        autorange: curve.value_kind === "mag" ? "reversed" : true,
      },
    };

      await Plotly.react(container.current, [trace], layout, {
      displayModeBar: false,
      responsive: true,
      });

      // Last line of defence. `webglSupport()` above answers "can this machine
      // make a WebGL context at all", which is not the same question as "will
      // regl succeed in this document right now" -- in the packaged build the
      // probe reports yes and regl still fails, so the grey placeholder gets
      // drawn over real data. Plotly reports that failure only as DOM (see
      // hasNoWebglPlaceholder), so detect it and redraw once through the SVG
      // path, which needs no WebGL and always renders.
      if (
        !disposed && container.current
        && trace.type === "scattergl"
        && hasNoWebglPlaceholder(container.current)
      ) {
        await Plotly.react(container.current, [{ ...trace, type: "scatter" }], layout, {
          displayModeBar: false,
          responsive: true,
        });
      }
    };
    // Plotly's scattergl trace needs a WebGL context. Losing one (GPU driver
    // reset, too many contexts already open, an unavailable GPU) throws
    // inside this async function; without a .catch() that becomes an
    // unhandled promise rejection and the plot silently stays blank with
    // only a console error -- indistinguishable from "the app just broke".
    render().catch((err: unknown) => {
      if (!disposed) setRenderError(err instanceof Error ? err.message : String(err));
    });
    return () => { disposed = true; };
  }, [curve, folded, resolved]);

  useEffect(() => {
    const node = container.current;
    return () => {
      void import("plotly.js-dist-min").then(({ default: Plotly }) => {
        if (node) Plotly.purge(node);
      });
    };
  }, []);

  if (renderError) {
    return (
      <div className="flex h-72 w-full flex-col items-center justify-center gap-1 rounded border border-[var(--color-edge)] text-xs text-[var(--color-muted)]">
        <p>Couldn&rsquo;t render this plot.</p>
        <p className="max-w-xs truncate text-[var(--color-bad)]" title={renderError}>
          {renderError}
        </p>
      </div>
    );
  }

  return <div ref={container} className="h-72 w-full" />;
}

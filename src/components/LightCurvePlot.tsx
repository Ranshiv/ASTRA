import { useEffect, useRef, useState } from "react";
import type Plotly from "plotly.js-dist-min";

import type { CurvePayload, FoldedCurve } from "@/lib/engine";
import { readThemeColor, useTheme } from "@/lib/theme";

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

/** `scattergl` needs a WebGL context. Where one isn't available (GPU
 * acceleration off in the webview, no compatible driver, too many contexts
 * already open), Plotly doesn't throw or fall back on its own -- it draws a
 * "WebGL is not supported" placeholder *inside* the plot area, which our
 * render() try/catch never sees because nothing actually failed. Checked
 * once per module load rather than per render: a WebGL context, once
 * unavailable, doesn't become available again mid-session. */
let webglSupported: boolean | null = null;
function supportsWebGL(): boolean {
  if (webglSupported !== null) return webglSupported;
  try {
    const canvas = document.createElement("canvas");
    webglSupported = !!(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    webglSupported = false;
  }
  return webglSupported;
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
      type: supportsWebGL() ? "scattergl" : "scatter",
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

      Plotly.react(container.current, [trace], layout, {
      displayModeBar: false,
      responsive: true,
      });
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

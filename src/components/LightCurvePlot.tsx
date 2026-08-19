import { useEffect, useRef, useState } from "react";
import type Plotly from "plotly.js-dist-min";

import type { CurvePayload, FoldedCurve } from "@/lib/engine";

const BAND_COLORS: Record<string, string> = {
  g: "#4ade80",
  r: "#f87171",
  i: "#fbbf24",
  TESS: "#6ea8ff",
  G: "#d8dce8",
};

/** Convert a #rrggbb colour into rgba() so error bars can be faded. */
function withAlpha(hex: string, alpha: number): string {
  const value = parseInt(hex.slice(1), 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

const LAYOUT_BASE: Partial<Plotly.Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#6b7590", size: 11 },
  margin: { l: 56, r: 16, t: 8, b: 40 },
  showlegend: false,
  hovermode: "closest",
};

const AXIS_BASE: Partial<Plotly.LayoutAxis> = {
  gridcolor: "#1c2030",
  zerolinecolor: "#1c2030",
  linecolor: "#1c2030",
};

export function LightCurvePlot({
  curve,
  folded,
}: {
  curve: CurvePayload;
  folded: FoldedCurve | null;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    if (!container.current) return;
    setRenderError(null);

    let disposed = false;
    const render = async () => {
      const { default: Plotly } = await import("plotly.js-dist-min");
      if (disposed || !container.current) return;
      const color = BAND_COLORS[curve.band] ?? "#6ea8ff";
    const showing = folded ?? curve;
    const x = folded ? folded.phase : curve.time;
    const y = showing.value;

    const trace: Partial<Plotly.PlotData> = {
      x,
      y,
      type: "scattergl",
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
      ...LAYOUT_BASE,
      xaxis: {
        ...AXIS_BASE,
        title: {
          text: folded ? "Phase" : `Time (${curve.time_system})`,
          font: { size: 11 },
        },
      },
      yaxis: {
        ...AXIS_BASE,
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
  }, [curve, folded]);

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

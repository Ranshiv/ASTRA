import { useEffect, useId, useRef, useState } from "react";

export interface SkyMarker {
  ra: number;
  dec: number;
  label?: string;
}

/** Aladin Lite sky view.
 *
 * The instance is created once and then steered with `gotoRaDec`, rather than
 * being torn down and rebuilt whenever the target moves. Rebuilding would drop
 * the tile cache and re-fetch every HiPS tile on each pan, which is slow and,
 * offline, turns a working map into a blank one.
 */
export function AladinSky({
  ra,
  dec,
  fov = 0.25,
  markers = [],
  height = "h-64",
}: {
  ra: number;
  dec: number;
  fov?: number;
  markers?: SkyMarker[];
  height?: string;
}) {
  const rawId = useId();
  const id = `aladin-${rawId.replace(/:/g, "")}`;
  const [status, setStatus] = useState("Loading sky map…");
  const aladin = useRef<any>(null);
  const api = useRef<any>(null);
  const overlay = useRef<any>(null);

  useEffect(() => {
    let disposed = false;
    // A webview without a 2D canvas (including jsdom and some remote/offline
    // shells) cannot initialize Aladin. Fail fast so the coordinate fallback
    // is useful instead of waiting on a large third-party module to hang.
    if (typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent)) {
      setStatus("Sky map unavailable offline; coordinates are shown below.");
      return () => { disposed = true; };
    }
    try {
      const canvas = document.createElement("canvas");
      if (!canvas.getContext("2d")) {
        setStatus("Sky map unavailable offline; coordinates are shown below.");
        return () => { disposed = true; };
      }
    } catch {
      setStatus("Sky map unavailable offline; coordinates are shown below.");
      return () => { disposed = true; };
    }
    let timedOut = false;
    const timeout = window.setTimeout(() => {
      timedOut = true;
      if (!disposed) setStatus("Sky map unavailable offline; coordinates are shown below.");
    }, 1500);
    void import("aladin-lite")
      .then(async ({ default: A }) => {
        await A.init;
        if (disposed || timedOut) return;
        api.current = A;
        aladin.current = A.aladin(`#${id}`, {
          target: `${ra} ${dec}`,
          fov,
          // TAN (tangential/gnomonic) is the pointed, narrow-field
          // projection -- the one an actual telescope image uses -- so it
          // crops to roughly `fov` degrees around the target. AIT (Aitoff)
          // and SIN ("Spheric" in this library -- an orthographic 3D globe,
          // not a zoomed rectangle) are both whole-sky-family projections:
          // they always render most or all of the celestial sphere and just
          // reposition the target within it, so `fov` has no real cropping
          // effect with either one.
          projection: "TAN",
          cooFrame: "equatorial",
          survey: "P/DSS2/color",
          // Aladin's default chrome is built for a full-page viewer, not a
          // small embedded panel: a 7-option projection dropdown, an ICRS
          // frame selector, and a big coordinate-readout box all fight for
          // space and clash with the app's dark theme. Coordinates are
          // already shown below the widget, so the readout is redundant;
          // the rest is control this app never needs. Zoom and fullscreen
          // are the only controls worth keeping in this footprint.
          showFrame: false,
          showCooLocation: false,
          showProjectionControl: false,
          showLayersControl: false,
          showSettingsControl: false,
          showSimbadPointerControl: false,
          showZoomControl: true,
          showFullscreenControl: true,
          showFov: true,
          showCooGrid: true,
          // Aladin's own defaults are a bright magenta (#ff54ff) grid and
          // reticle, which reads as an error/alert colour against this
          // theme rather than a coordinate overlay. Matched to the app's
          // accent colour instead, at low opacity so it stays a guide, not
          // the most prominent thing on screen.
          gridColor: "#6ea8ff",
          gridOpacity: 0.35,
          reticleColor: "#6ea8ff",
        });
        window.clearTimeout(timeout);
        setStatus("");
      })
      .catch(() => {
        window.clearTimeout(timeout);
        if (!disposed && !timedOut) setStatus("Sky map unavailable offline; coordinates are shown below.");
      });
    return () => {
      disposed = true;
      window.clearTimeout(timeout);
    };
    // Created once: target and markers are applied by the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (!aladin.current) return;
    try {
      aladin.current.gotoRaDec(ra, dec);
      aladin.current.setFoV(fov);
    } catch {
      /* an offline map has no viewport to steer */
    }
  }, [ra, dec, fov, status]);

  useEffect(() => {
    if (!aladin.current || !api.current) return;
    try {
      if (overlay.current) aladin.current.removeLayer(overlay.current);
      if (markers.length === 0) {
        overlay.current = null;
        return;
      }
      const catalog = api.current.catalog({ name: "Stored sources", sourceSize: 10 });
      aladin.current.addCatalog(catalog);
      catalog.addSources(
        markers.map((marker) =>
          api.current.source(marker.ra, marker.dec, { label: marker.label ?? "" }),
        ),
      );
      overlay.current = catalog;
    } catch {
      /* overlays are a convenience; the map stays usable without them */
    }
  }, [markers, status]);

  return (
    <div>
      <div
        id={id}
        // Aladin positions its toolbar, zoom buttons and coordinate box with
        // `position: absolute` internally. Without an explicit positioning
        // context on this div, those elements resolve against the nearest
        // *actually* positioned ancestor -- or the viewport, if there isn't
        // one -- which is what let the widget's chrome bleed out over the
        // whole page instead of staying inside this box. `overflow-hidden`
        // alone does not fix this: it only clips descendants whose
        // containing block is this element, and without `relative` here,
        // Aladin's absolutely-positioned pieces have a different one.
        className={`astra-aladin relative ${height} w-full overflow-hidden rounded border border-[var(--color-edge)]`}
      />
      <p className="mt-1 text-[11px] text-[var(--color-muted)]">
        {status ||
          `${ra.toFixed(6)}°, ${dec.toFixed(6)}° · ${fov}° field${
            markers.length ? ` · ${markers.length} stored sources` : ""
          } · remote HiPS tiles`}
      </p>
    </div>
  );
}

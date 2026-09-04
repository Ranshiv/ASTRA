/** Shared WebGL capability probe.
 *
 * Three consumers need to know, up front, whether WebGL will actually work
 * before committing to a renderer that assumes it does: LightCurvePlot
 * (Plotly's `scattergl`, via regl), AladinSky (aladin-lite, a WebGL2-only
 * WASM renderer) and SpatialScene3D (three.js's WebGLRenderer). A bare
 * `canvas.getContext('webgl')` is not a reliable stand-in for "will regl/
 * aladin-lite/three actually initialize" -- a webview can hand back a context
 * object for an attribute set those libraries don't use, then fail once the
 * library's own (stricter) request comes in. That mismatch is what produced
 * Plotly's grey "WebGL is not supported" placeholder while this module's
 * predecessor (LightCurvePlot's old local `supportsWebGL`) reported success.
 *
 * Probed once per module load, not per render: in a given session a WebGL
 * context, once unavailable, doesn't become available again.
 */

export interface WebGLSupport {
  webgl1: boolean;
  webgl2: boolean;
  /** UNMASKED_RENDERER_WEBGL string, e.g. "ANGLE (Google, Vulkan)" or
   * "SwiftShader" -- null if unavailable or WEBGL_debug_renderer_info isn't
   * exposed. */
  renderer: string | null;
}

/** The attribute set regl (and therefore Plotly's `scattergl`) requests --
 * mirrored here so the probe fails exactly when regl would. */
const REGL_CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: true,
  antialias: true,
  preserveDrawingBuffer: true,
  premultipliedAlpha: false,
};

/** Extensions regl treats as mandatory; a context missing either one is not
 * one regl can actually use, even though `getContext` itself succeeded. */
const REGL_REQUIRED_EXTENSIONS = ["ANGLE_instanced_arrays", "OES_element_index_uint"];

function probeRenderer(gl: WebGLRenderingContext | WebGL2RenderingContext): string | null {
  try {
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    if (!info) return null;
    const value = gl.getParameter(info.UNMASKED_RENDERER_WEBGL);
    return typeof value === "string" ? value : null;
  } catch {
    return null;
  }
}

/** Releases a probe context immediately rather than leaving it live -- a
 * webview allows only a bounded number of concurrent WebGL contexts, and this
 * probe runs before the real consumer (Plotly/aladin-lite/three) claims its
 * own. */
function releaseContext(gl: WebGLRenderingContext | WebGL2RenderingContext): void {
  try {
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    /* best effort; the canvas is about to be garbage collected regardless */
  }
}

let cached: WebGLSupport | null = null;

export function webglSupport(): WebGLSupport {
  if (cached !== null) return cached;

  let webgl1 = false;
  let webgl2 = false;
  let renderer: string | null = null;

  try {
    const canvas = document.createElement("canvas");

    const gl2 = canvas.getContext("webgl2") as WebGL2RenderingContext | null;
    if (gl2) {
      webgl2 = true;
      renderer = probeRenderer(gl2);
      releaseContext(gl2);
    }

    // A fresh canvas for the WebGL1 probe: a canvas that has already vended a
    // webgl2 context cannot then also vend a webgl (v1) one.
    const canvas1 = document.createElement("canvas");
    const gl1 = canvas1.getContext("webgl", REGL_CONTEXT_ATTRIBUTES) as WebGLRenderingContext | null;
    if (gl1) {
      const hasRequiredExtensions = REGL_REQUIRED_EXTENSIONS.every((name) => !!gl1.getExtension(name));
      if (hasRequiredExtensions) {
        webgl1 = true;
        if (!renderer) renderer = probeRenderer(gl1);
      }
      releaseContext(gl1);
    }
  } catch {
    webgl1 = false;
    webgl2 = false;
    renderer = null;
  }

  cached = { webgl1, webgl2, renderer };
  return cached;
}

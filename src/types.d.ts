declare module "aladin-lite" {
  // aladin-lite ships no types of its own; this shim covers exactly the
  // surface AladinSky.tsx calls, not the full (undocumented, minified)
  // library API. `setFoV`/`setFov` are genuinely both real: the library's
  // own bundle defines `setFov` as a lowercase-v alias of `setFoV`.
  export interface AladinCatalog {
    addSources: (sources: unknown[]) => void;
  }

  export interface AladinInstance {
    gotoRaDec: (ra: number, dec: number) => void;
    setFoV: (fov: number) => void;
    setFov: (fov: number) => void;
    removeLayer: (layer: AladinCatalog) => void;
    addCatalog: (catalog: AladinCatalog) => void;
  }

  export interface AladinStatic {
    init: Promise<void>;
    aladin: (selector: string, options: Record<string, unknown>) => AladinInstance;
    catalog: (options: Record<string, unknown>) => AladinCatalog;
    source: (ra: number, dec: number, data?: Record<string, unknown>) => unknown;
  }

  const A: AladinStatic;
  export default A;
}

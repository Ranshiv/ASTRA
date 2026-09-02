import { useEffect, useRef, useState } from "react";
import type {
  BufferGeometry,
  Fog,
  LineBasicMaterial,
  MeshBasicMaterial,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  Vector2,
  WebGLRenderer,
} from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { SpatialCandidatePoint } from "@/lib/engine";
import { readThemeColorHexInt, useTheme } from "@/lib/theme";

/** Interactive 3D scatter of candidates by RA/Dec/Gaia distance.
 *
 * Mirrors AladinSky.tsx's discipline exactly: the renderer is created once in
 * an effect keyed only on mount, then steered by later effects when `points`
 * changes -- rebuilding the whole WebGL context on every prop change would
 * both be slow and lose camera position on every data refresh.
 *
 * Distance spans orders of magnitude (parsecs to kiloparsecs), so points are
 * placed on a log-scaled radius. A point with no reliable distance is never
 * plotted at a fabricated position -- it is simply absent, and the caller
 * reports how many were excluded and why (SpatialResult.reliable vs total).
 */

/** Read live so a theme change is reflected without rebuilding the WebGL
 * context. Previously a frozen object of Tailwind-500 shades that had
 * drifted from ui.tsx's actual Badge tones (only `accent` matched); sourcing
 * from the tokens fixes that drift as a side effect. */
function toneColors() {
  return {
    bad: readThemeColorHexInt("--color-bad"),
    warn: readThemeColorHexInt("--color-warn"),
    ok: readThemeColorHexInt("--color-ok"),
    accent: readThemeColorHexInt("--color-accent"),
  };
}

function toneForScore(score: number | null, tones: ReturnType<typeof toneColors>): number {
  if (score === null || !Number.isFinite(score)) return tones.accent;
  if (score >= 0.7) return tones.bad; // highest anomaly scores stand out
  if (score >= 0.4) return tones.warn;
  return tones.ok;
}

// log10(pc) typically spans ~1 (10 pc) to ~4 (10 kpc); shifted and scaled
// into a camera-friendly range rather than plotting raw parsecs, which
// would crush every nearby point into the origin.
function sceneRadiusForDistance(distancePc: number): number {
  return Math.max(Math.log10(Math.max(distancePc, 1)) - 1, 0.1) * 15;
}

/** Round parsec values marked by the wireframe distance shells, nearest to
 * farthest -- also drives the equator ring's radius (the farthest shell). */
const DISTANCE_SHELLS_PC = [100, 1000, 10000] as const;

/** RA/Dec/parsecs -> a scene-scale Cartesian position, log-radius. */
function toSceneXYZ(raDeg: number, decDeg: number, distancePc: number): [number, number, number] {
  const ra = (raDeg * Math.PI) / 180;
  const dec = (decDeg * Math.PI) / 180;
  const radius = sceneRadiusForDistance(distancePc);
  return [
    radius * Math.cos(dec) * Math.cos(ra),
    radius * Math.cos(dec) * Math.sin(ra),
    radius * Math.sin(dec),
  ];
}

export function SpatialScene3D({
  points,
  loading = false,
  onSelect,
  height = "h-64",
}: {
  points: SpatialCandidatePoint[];
  /** True while the candidate/distance data itself is still being fetched,
   * distinct from `status` below (which tracks the WebGL scene's own
   * one-time setup) -- without this, an empty result and a still-loading
   * result render identically as "0 of 0 plotted". */
  loading?: boolean;
  onSelect?: (candidateId: string) => void;
  height?: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState("Loading 3D view…");
  const sceneApi = useRef<{
    scene: Scene;
    camera: PerspectiveCamera;
    renderer: WebGLRenderer;
    controls: OrbitControls;
    points: Points<BufferGeometry, PointsMaterial>;
    raycaster: Raycaster;
    pointer: Vector2;
    plotted: SpatialCandidatePoint[];
    frameId: number;
    // Kept so a later theme-change effect can restyle these in place rather
    // than rebuilding the WebGL context (which would lose camera position).
    fog: Fog;
    anchorMaterial: MeshBasicMaterial;
    shellMaterials: LineBasicMaterial[];
    equatorMaterial: LineBasicMaterial;
  } | null>(null);
  const { resolved } = useTheme();

  // Created once. Later effects steer this instance rather than rebuilding
  // it, so orbit position survives a data refresh.
  useEffect(() => {
    let disposed = false;
    const container = containerRef.current;
    if (!container) return;

    void Promise.all([
      import("three"),
      import("three/examples/jsm/controls/OrbitControls.js"),
    ])
      .then(([THREE, { OrbitControls }]) => {
        if (disposed || !container) return;

        const tones = toneColors();
        const voidColor = readThemeColorHexInt("--color-void");
        const edgeColor = readThemeColorHexInt("--color-edge");
        const mutedColor = readThemeColorHexInt("--color-muted");

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(voidColor);
        // Points near the fog's far edge fade into the background instead of
        // staying tack-sharp -- without this, orbiting gives almost no sense
        // of depth since every point renders at the same apparent contrast
        // regardless of distance from the camera.
        const fog = new THREE.Fog(voidColor, 30, 160);
        scene.fog = fog;

        // Every candidate's position is relative to this point (direction +
        // log-distance from Earth); without a visible anchor there, the
        // layout has nothing for the eye to read depth against.
        const anchorMaterial = new THREE.MeshBasicMaterial({ color: tones.accent });
        scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), anchorMaterial));

        // Concentric distance-shell wireframes at round parsec values. Their
        // near/far arcs move at different apparent rates while orbiting,
        // which is what actually reads as "3D" -- flat dots on black don't,
        // regardless of how the camera is set up.
        const shellMaterials: LineBasicMaterial[] = [];
        for (const distancePc of DISTANCE_SHELLS_PC) {
          const wireframe = new THREE.WireframeGeometry(
            new THREE.SphereGeometry(sceneRadiusForDistance(distancePc), 16, 10),
          );
          const shellMaterial = new THREE.LineBasicMaterial({
            color: edgeColor, transparent: true, opacity: 0.07,
          });
          scene.add(new THREE.LineSegments(wireframe, shellMaterial));
          shellMaterials.push(shellMaterial);
        }

        // Celestial-equator ring (Dec = 0) at the outermost shell's radius,
        // so orbiting also conveys orientation, not just depth.
        const equatorRadius = sceneRadiusForDistance(
          DISTANCE_SHELLS_PC[DISTANCE_SHELLS_PC.length - 1]);
        const equatorPoints = Array.from({ length: 65 }, (_, i) => {
          const t = (i / 64) * Math.PI * 2;
          return new THREE.Vector3(equatorRadius * Math.cos(t), equatorRadius * Math.sin(t), 0);
        });
        const equatorMaterial = new THREE.LineBasicMaterial({
          color: mutedColor, transparent: true, opacity: 0.12,
        });
        scene.add(new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(equatorPoints),
          equatorMaterial,
        ));

        const width = container.clientWidth || 400;
        const heightPx = container.clientHeight || 300;
        const camera = new THREE.PerspectiveCamera(50, width / heightPx, 0.1, 2000);
        camera.position.set(40, 40, 40);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, heightPx);
        container.replaceChildren(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({
          size: 1.4, vertexColors: true, sizeAttenuation: true,
        });
        const cloud = new THREE.Points(geometry, material);
        scene.add(cloud);

        const raycaster = new THREE.Raycaster();
        raycaster.params.Points = { threshold: 1.5 };

        function animate() {
          controls.update();
          renderer.render(scene, camera);
          if (sceneApi.current) sceneApi.current.frameId = requestAnimationFrame(animate);
        }

        sceneApi.current = {
          scene, camera, renderer, controls, points: cloud, raycaster,
          pointer: new THREE.Vector2(), plotted: [], frameId: 0,
          fog, anchorMaterial, shellMaterials, equatorMaterial,
        };
        animate();
        setStatus("");
      })
      .catch(() => setStatus("3D view unavailable offline."));

    return () => {
      disposed = true;
      const api = sceneApi.current;
      if (api) {
        cancelAnimationFrame(api.frameId);
        api.controls.dispose();
        api.renderer.dispose();
      }
      sceneApi.current = null;
    };
  }, []);

  // Steer geometry/colors whenever the data changes, without rebuilding the
  // renderer -- same discipline as AladinSky's marker-overlay effect.
  useEffect(() => {
    const api = sceneApi.current;
    if (!api) return;

    void import("three").then((THREE) => {
      const plotted = points.filter(
        (p) => p.distance_reliable && p.gaia_distance_pc !== null,
      );
      const positions = new Float32Array(plotted.length * 3);
      const colors = new Float32Array(plotted.length * 3);
      const color = new THREE.Color();
      const tones = toneColors();

      plotted.forEach((point, index) => {
        const [x, y, z] = toSceneXYZ(point.ra_deg, point.dec_deg, point.gaia_distance_pc as number);
        positions.set([x, y, z], index * 3);
        color.setHex(toneForScore(point.score_total, tones));
        colors.set([color.r, color.g, color.b], index * 3);
      });

      api.points.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      api.points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      api.points.geometry.computeBoundingSphere();
      api.plotted = plotted;
    });
  }, [points, status, resolved]);

  // Restyle the fixed scene chrome (background, fog, anchor, distance
  // shells, equator ring) on a theme change, without touching the WebGL
  // context itself -- same "steer, don't rebuild" discipline as the effect
  // above and AladinSky's marker overlay.
  useEffect(() => {
    const api = sceneApi.current;
    if (!api) return;
    void import("three").then((THREE) => {
      const tones = toneColors();
      const voidColor = readThemeColorHexInt("--color-void");
      const edgeColor = readThemeColorHexInt("--color-edge");
      const mutedColor = readThemeColorHexInt("--color-muted");
      api.scene.background = new THREE.Color(voidColor);
      api.fog.color.setHex(voidColor);
      api.anchorMaterial.color.setHex(tones.accent);
      for (const shellMaterial of api.shellMaterials) shellMaterial.color.setHex(edgeColor);
      api.equatorMaterial.color.setHex(mutedColor);
    });
  }, [resolved, status]);

  useEffect(() => {
    const container = containerRef.current;
    const api = sceneApi.current;
    if (!container || !api || !onSelect) return;

    function handleClick(event: MouseEvent) {
      const rect = container!.getBoundingClientRect();
      api!.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      api!.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      api!.raycaster.setFromCamera(api!.pointer, api!.camera);
      const hits = api!.raycaster.intersectObject(api!.points);
      if (hits.length > 0) {
        const point = api!.plotted[hits[0].index as number];
        if (point) onSelect!(point.candidate_id);
      }
    }

    container.addEventListener("click", handleClick);
    return () => container.removeEventListener("click", handleClick);
  }, [onSelect, points]);

  const plottedCount = points.filter((p) => p.distance_reliable).length;

  return (
    <div>
      <div
        ref={containerRef}
        className={`relative ${height} w-full overflow-hidden rounded border border-[var(--color-edge)]`}
      />
      <p className="mt-1 text-[11px] text-[var(--color-muted)]">
        {status ||
          (loading
            ? "Loading candidates…"
            : `${plottedCount} of ${points.length} plotted · drag to orbit, scroll to zoom`)}
      </p>
      {!status && !loading && (
        <p className="text-[11px] text-[var(--color-muted)]">
          Center marker is Earth; rings are 100 pc, 1 kpc, and 10 kpc from it.
        </p>
      )}
    </div>
  );
}

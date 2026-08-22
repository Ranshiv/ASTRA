import { useEffect, useRef, useState } from "react";

import type { SpatialCandidatePoint } from "@/lib/engine";

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

const TONE_COLORS = {
  // Matches ui.tsx's Badge tone palette, translated to RGB for WebGL.
  bad: 0xef4444,
  warn: 0xf59e0b,
  ok: 0x34d399,
  accent: 0x6ea8ff,
} as const;

function toneForScore(score: number | null): number {
  if (score === null || !Number.isFinite(score)) return TONE_COLORS.accent;
  if (score >= 0.7) return TONE_COLORS.bad; // highest anomaly scores stand out
  if (score >= 0.4) return TONE_COLORS.warn;
  return TONE_COLORS.ok;
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
    scene: any;
    camera: any;
    renderer: any;
    controls: any;
    points: any;
    raycaster: any;
    pointer: any;
    plotted: SpatialCandidatePoint[];
    frameId: number;
  } | null>(null);

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

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e14);
        // Points near the fog's far edge fade into the background instead of
        // staying tack-sharp -- without this, orbiting gives almost no sense
        // of depth since every point renders at the same apparent contrast
        // regardless of distance from the camera.
        scene.fog = new THREE.Fog(0x0a0e14, 30, 160);

        // Every candidate's position is relative to this point (direction +
        // log-distance from Earth); without a visible anchor there, the
        // layout has nothing for the eye to read depth against.
        scene.add(new THREE.Mesh(
          new THREE.SphereGeometry(0.6, 16, 16),
          new THREE.MeshBasicMaterial({ color: TONE_COLORS.accent }),
        ));

        // Concentric distance-shell wireframes at round parsec values. Their
        // near/far arcs move at different apparent rates while orbiting,
        // which is what actually reads as "3D" -- flat dots on black don't,
        // regardless of how the camera is set up.
        for (const distancePc of DISTANCE_SHELLS_PC) {
          const wireframe = new THREE.WireframeGeometry(
            new THREE.SphereGeometry(sceneRadiusForDistance(distancePc), 16, 10),
          );
          scene.add(new THREE.LineSegments(
            wireframe,
            new THREE.LineBasicMaterial({ color: 0x2a3350, transparent: true, opacity: 0.07 }),
          ));
        }

        // Celestial-equator ring (Dec = 0) at the outermost shell's radius,
        // so orbiting also conveys orientation, not just depth.
        const equatorRadius = sceneRadiusForDistance(
          DISTANCE_SHELLS_PC[DISTANCE_SHELLS_PC.length - 1]);
        const equatorPoints = Array.from({ length: 65 }, (_, i) => {
          const t = (i / 64) * Math.PI * 2;
          return new THREE.Vector3(equatorRadius * Math.cos(t), equatorRadius * Math.sin(t), 0);
        });
        scene.add(new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(equatorPoints),
          new THREE.LineBasicMaterial({ color: 0x3a4570, transparent: true, opacity: 0.12 }),
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

      plotted.forEach((point, index) => {
        const [x, y, z] = toSceneXYZ(point.ra_deg, point.dec_deg, point.gaia_distance_pc as number);
        positions.set([x, y, z], index * 3);
        color.setHex(toneForScore(point.score_total));
        colors.set([color.r, color.g, color.b], index * 3);
      });

      api.points.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      api.points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      api.points.geometry.computeBoundingSphere();
      api.plotted = plotted;
    });
  }, [points, status]);

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

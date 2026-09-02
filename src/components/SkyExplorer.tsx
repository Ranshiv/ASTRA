/** Standalone sky explorer (plan section 10).
 *
 * Aladin was previously only reachable inside a candidate's detail pane, so
 * there was no way to look at a region before deciding what to acquire. Stored
 * candidate positions are overlaid so the map shows what ASTRA actually holds,
 * not just what the sky contains.
 *
 * The time-frame offset is shown here rather than buried in a settings screen
 * because it is a property of the POSITION: it is what ASTRA subtracts to put
 * every survey on one BJD_TDB axis, and it varies with where you are looking.
 */
import { Box, Clock, Compass, Crosshair, Globe } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { AladinSky, type SkyMarker } from "@/components/AladinSky";
import { SpatialScene3D } from "@/components/SpatialScene3D";
import { engine, type Candidate, type FrameOffset } from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, num, useAsync } from "@/components/ui";

export function SkyExplorer({ projectId }: { projectId?: string }) {
  const [ra, setRa] = useState("180.122");
  const [dec, setDec] = useState("22.411");
  const [fov, setFov] = useState("0.5");
  // Whether the user has manually steered ra/dec/fov since the last project
  // switch. While untouched, the auto-fit effect below is free to snap the
  // view to wherever the loaded candidates actually are -- a multi-region
  // mosaic's markers can spread far wider than these static defaults, and
  // without this the sky view stays zoomed into a near-empty patch even
  // though the project's data covers the whole visible sky. Once the user
  // moves the view manually, their choice is respected instead of being
  // silently overridden on the next data reload.
  const [touched, setTouched] = useState(false);
  const [offset, setOffset] = useState<FrameOffset | null>(null);
  const [offsetError, setOffsetError] = useState<string | null>(null);
  const [measuringOffset, setMeasuringOffset] = useState(false);
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

  function goTo(nextRa: number, nextDec: number) {
    setTouched(true);
    setRa(String(nextRa));
    setDec(String(nextDec));
  }

  const { data: candidates, error, loading: candidatesLoading } = useAsync(
    () => engine.candidates("default", 200, projectId), [projectId]);
  const spatial = useAsync(
    () => engine.candidatesSpatial("default", 200, projectId), [projectId]);

  const raDeg = Number(ra);
  const decDeg = Number(dec);
  // `Number("")` is `0`, not NaN -- without the blank checks a cleared field
  // reads as a valid position at 0/0 instead of disabling the view.
  const valid = ra.trim() !== "" && dec.trim() !== ""
    && Number.isFinite(raDeg) && Number.isFinite(decDeg)
    && raDeg >= 0 && raDeg < 360 && decDeg >= -90 && decDeg <= 90;

  const markers: SkyMarker[] = useMemo(
    () =>
      (candidates?.candidates ?? []).map((candidate: Candidate) => ({
        ra: candidate.ra_deg,
        dec: candidate.dec_deg,
        label: candidate.candidate_id,
      })),
    [candidates],
  );

  // A new project's data may spread across several acquired regions, well
  // outside the static "180.122 / 22.411 / 0.5" defaults above -- snap the
  // view to fit it once, and only while the user hasn't steered manually.
  useEffect(() => {
    setTouched(false);
  }, [projectId]);

  useEffect(() => {
    if (touched || markers.length === 0) return;
    const ras = markers.map((m) => m.ra);
    const decs = markers.map((m) => m.dec);
    const raMin = Math.min(...ras);
    const raMax = Math.max(...ras);
    const decMin = Math.min(...decs);
    const decMax = Math.max(...decs);
    const spanDeg = Math.max(raMax - raMin, decMax - decMin);
    setRa((((raMin + raMax) / 2)).toFixed(6));
    setDec((((decMin + decMax) / 2)).toFixed(6));
    setFov(Math.max(spanDeg * 1.3, 0.1).toFixed(4));
    // Intentionally excludes `touched`: this must run once per markers load
    // while untouched, not re-fire every time `touched` itself changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markers]);

  async function measureOffset() {
    setOffsetError(null);
    setMeasuringOffset(true);
    try {
      setOffset(await engine.frameOffset(raDeg, decDeg));
    } catch (err) {
      setOffsetError(String(err));
    } finally {
      setMeasuringOffset(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Compass}
        title="Sky explorer"
        description="Point the map anywhere; stored candidate positions are overlaid."
        actions={
          <>
            <Button
              icon={Globe}
              tone={viewMode === "2d" ? "accent" : "default"}
              onClick={() => setViewMode("2d")}
            >
              2D
            </Button>
            <Button
              icon={Box}
              tone={viewMode === "3d" ? "accent" : "default"}
              onClick={() => setViewMode("3d")}
            >
              3D
            </Button>
            <Button
              icon={Clock}
              disabled={!valid || measuringOffset}
              loading={measuringOffset}
              onClick={() => void measureOffset()}
            >
              Time-frame offset
            </Button>
            <Badge tone="muted">
              {candidatesLoading ? "Loading…" : `${markers.length} stored candidates`}
            </Badge>
          </>
        }
      >
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <Field label="RA (deg)" value={ra} onChange={(v) => { setTouched(true); setRa(v); }} min={0} max={359.999999} />
          <Field label="Dec (deg)" value={dec} onChange={(v) => { setTouched(true); setDec(v); }} min={-90} max={90} />
          <Field
            label="Field of view (deg)"
            value={fov}
            onChange={(v) => { setTouched(true); setFov(v); }}
            width="w-24"
            min={0.01}
            max={180}
            step="any"
          />
        </div>

        {!valid ? (
          <Note tone="warn">Enter numeric coordinates in degrees.</Note>
        ) : viewMode === "2d" ? (
          <AladinSky
            ra={raDeg}
            dec={decDeg}
            fov={Number(fov) || 0.5}
            markers={markers}
            height="h-[min(28rem,55vh)]"
          />
        ) : (
          <>
            {spatial.error && <Note tone="bad">{spatial.error}</Note>}
            <SpatialScene3D
              points={spatial.data?.points ?? []}
              loading={spatial.loading}
              onSelect={(candidateId) => {
                const target = candidates?.candidates.find(
                  (c) => c.candidate_id === candidateId,
                );
                if (target) goTo(target.ra_deg, target.dec_deg);
              }}
              height="h-[min(28rem,55vh)]"
            />
            {spatial.data && (
              <Note tone={spatial.data.reliable < spatial.data.total ? "warn" : "muted"}>
                {spatial.data.reliable} of {spatial.data.total} candidates plotted;{" "}
                {spatial.data.total - spatial.data.reliable} lack a reliable Gaia distance
                (parallax SNR &lt; {spatial.data.snr_threshold} or no match).
              </Note>
            )}
          </>
        )}

        {offsetError && <Note tone="bad">{offsetError}</Note>}
        {offset && (
          <div className="mt-3 rounded-lg border border-[var(--color-accent)]/40 bg-[var(--color-accent)]/5 p-3">
            <h3 className="mb-1.5 text-xs font-medium text-[var(--color-text)]">
              HJD_UTC → BJD_TDB at this position
            </h3>
            <KeyValue
              rows={[
                ["UTC → TDB scale", `${num(offset.scale_seconds, 2)} s`],
                ["Heliocentric → barycentric", `${num(offset.reference_seconds, 2)} s`],
                ["Total", `${num(offset.total_seconds, 2)} s`],
              ]}
            />
            <Note>
              About a minute, not the ±8.3 minutes often quoted — that figure is the
              geocentric-to-barycentric term, which HJD has already absorbed. Still worth
              correcting: it is over half a TESS two-minute cadence bin.
            </Note>
          </div>
        )}
      </Panel>

      <Panel
        icon={Crosshair}
        title="Stored candidates"
        description="Select one to centre the map on it."
      >
        {error && <Note tone="bad">{error}</Note>}
        {candidatesLoading ? (
          <Note>Loading stored candidates…</Note>
        ) : markers.length === 0 ? (
          <Empty>No candidates stored yet. Build a candidate run first.</Empty>
        ) : (
          <div className="flex max-h-64 flex-wrap gap-1.5 overflow-y-auto">
            {(candidates?.candidates ?? []).map((candidate) => (
              <button
                key={candidate.candidate_id}
                type="button"
                onClick={() => goTo(candidate.ra_deg, candidate.dec_deg)}
                aria-label={`Centre map on candidate ${candidate.candidate_id}`}
                className="min-h-9 rounded-full border border-[var(--color-edge)] px-3 py-1 font-mono text-xs text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-void)]"
              >
                #{candidate.rank} {candidate.candidate_id}
              </button>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

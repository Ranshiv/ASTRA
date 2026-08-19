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
import { Clock, Compass, Crosshair } from "lucide-react";
import { useMemo, useState } from "react";

import { AladinSky, type SkyMarker } from "@/components/AladinSky";
import { engine, type Candidate, type FrameOffset } from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, num, useAsync } from "@/components/ui";

export function SkyExplorer({ projectId }: { projectId?: string }) {
  const [ra, setRa] = useState("180.122");
  const [dec, setDec] = useState("22.411");
  const [fov, setFov] = useState("0.5");
  const [offset, setOffset] = useState<FrameOffset | null>(null);
  const [offsetError, setOffsetError] = useState<string | null>(null);

  const { data: candidates, error } = useAsync(() => engine.candidates("default", 200, projectId));

  const raDeg = Number(ra);
  const decDeg = Number(dec);
  const valid = Number.isFinite(raDeg) && Number.isFinite(decDeg);

  const markers: SkyMarker[] = useMemo(
    () =>
      (candidates?.candidates ?? []).map((candidate: Candidate) => ({
        ra: candidate.ra_deg,
        dec: candidate.dec_deg,
        label: candidate.candidate_id,
      })),
    [candidates],
  );

  async function measureOffset() {
    setOffsetError(null);
    try {
      setOffset(await engine.frameOffset(raDeg, decDeg));
    } catch (err) {
      setOffsetError(String(err));
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
            <Button icon={Clock} disabled={!valid} onClick={() => void measureOffset()}>
              Time-frame offset
            </Button>
            <Badge tone="muted">{markers.length} stored candidates</Badge>
          </>
        }
      >
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <Field label="RA (deg)" value={ra} onChange={setRa} />
          <Field label="Dec (deg)" value={dec} onChange={setDec} />
          <Field label="Field of view (deg)" value={fov} onChange={setFov} width="w-24" />
        </div>

        {!valid ? (
          <Note tone="warn">Enter numeric coordinates in degrees.</Note>
        ) : (
          <AladinSky
            ra={raDeg}
            dec={decDeg}
            fov={Number(fov) || 0.5}
            markers={markers}
            height="h-[28rem]"
          />
        )}

        {offsetError && <Note tone="bad">{offsetError}</Note>}
        {offset && (
          <div className="mt-3">
            <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">
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
        {markers.length === 0 ? (
          <Empty>No candidates stored yet. Build a candidate run first.</Empty>
        ) : (
          <div className="flex max-h-64 flex-wrap gap-1.5 overflow-y-auto">
            {(candidates?.candidates ?? []).map((candidate) => (
              <button
                key={candidate.candidate_id}
                type="button"
                onClick={() => {
                  setRa(String(candidate.ra_deg));
                  setDec(String(candidate.dec_deg));
                }}
                className="rounded-full border border-[var(--color-edge)] px-2.5 py-1 font-mono text-[11px] text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
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

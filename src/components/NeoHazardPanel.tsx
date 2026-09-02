/** NEO hazard quantities -- MOID, Tisserand parameter, absolute magnitude,
 *  PHA classification, and close-approach distance -- built on
 *  `moving_objects.py`'s existing orbit machinery (roadmap: astrophysics
 *  & extraterrestrial-study feature pass).
 *
 *  `neo.assess`/`neo.close_approach` are read-only diagnostic RPC methods,
 *  the same category `physical.characterize`/`digital_twin.*` occupy.
 *  This engine has no impact-probability model -- see `neo_hazard.py`'s
 *  own `[GAP]` -- so a close-approach distance is shown alongside its
 *  own explicit caveat, never framed as an impact forecast.
 */
import { useState } from "react";
import { Orbit, Radar } from "lucide-react";

import {
  engine,
  type NeoCloseApproach,
  type NeoHazardAssessment,
  type OrbitalElements,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, num } from "@/components/ui";

function AssessmentDetail({ result }: { result: NeoHazardAssessment }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result.is_pha ? "bad" : "muted"}>
          {result.is_pha ? "PHA (MOID <= 0.05 AU and H <= 22.0)" : "not a PHA"}
        </Badge>
        <Badge tone="muted">{result.dynamical_class}</Badge>
      </div>
      <KeyValue
        rows={[
          ["MOID (AU)", num(result.moid_au, 5)],
          ["Tisserand parameter (Jupiter)", num(result.tisserand_jupiter, 3)],
          ["Absolute magnitude H", num(result.absolute_magnitude, 2)],
          ["Diameter (km, p_V=0.14)", num(result.diameter_km, 3)],
          [
            "Diameter range (km, p_V 0.05-0.25)",
            result.diameter_km_range
              ? `${num(result.diameter_km_range[0], 3)} - ${num(result.diameter_km_range[1], 3)}`
              : "—",
          ],
        ]}
      />
      <Note tone="warn">
        MOID is a numerical snapshot at the given osculating elements (no secular
        precession), and close-approach/orbit propagation here is unperturbed two-body
        with no covariance. This engine reports no impact probability.
      </Note>
    </div>
  );
}

function CloseApproachDetail({ result }: { result: NeoCloseApproach }) {
  return (
    <div className="flex flex-col gap-3">
      <KeyValue
        rows={[
          ["Closest approach (MJD)", num(result.close_approach_mjd, 3)],
          ["Distance (AU)", num(result.distance_au, 5)],
          ["Distance (lunar distances)", num(result.distance_lunar_distances, 2)],
          ["Window (MJD)", `${num(result.window_start_mjd, 1)} - ${num(result.window_end_mjd, 1)}`],
        ]}
      />
      <Note tone="warn">
        Unperturbed two-body propagation only -- no planetary perturbations, no
        Yarkovsky effect, no propagated uncertainty. This is a distance, not a
        collision-risk assessment.
      </Note>
    </div>
  );
}

const DEFAULT_ELEMENTS: OrbitalElements = {
  semi_major_axis_au: 1.5, eccentricity: 0.3, inclination_deg: 10.0,
  raan_deg: 0.0, argument_of_perihelion_deg: 0.0, mean_anomaly_deg: 0.0, epoch_mjd: 60000.0,
};

function ElementFields({ elements, onChange }: {
  elements: OrbitalElements;
  onChange: (next: OrbitalElements) => void;
}) {
  const set = (key: keyof OrbitalElements) => (value: string) =>
    onChange({ ...elements, [key]: Number(value) || 0 });
  return (
    <div className="flex flex-wrap items-end gap-2">
      <Field label="a (AU)" value={String(elements.semi_major_axis_au)} onChange={set("semi_major_axis_au")} width="w-20" />
      <Field label="e" value={String(elements.eccentricity)} onChange={set("eccentricity")} width="w-16" />
      <Field label="i (deg)" value={String(elements.inclination_deg)} onChange={set("inclination_deg")} width="w-20" />
    </div>
  );
}

export function NeoHazardPanel({ projectId: _projectId }: { projectId?: string }) {
  const [elements, setElements] = useState<OrbitalElements>(DEFAULT_ELEMENTS);
  const [assessment, setAssessment] = useState<NeoHazardAssessment | null>(null);
  const [assessBusy, setAssessBusy] = useState(false);
  const [assessError, setAssessError] = useState<string | null>(null);

  const [startMjd, setStartMjd] = useState("60000");
  const [endMjd, setEndMjd] = useState("60400");
  const [closeApproach, setCloseApproach] = useState<NeoCloseApproach | null>(null);
  const [approachBusy, setApproachBusy] = useState(false);
  const [approachError, setApproachError] = useState<string | null>(null);

  async function runAssessment() {
    setAssessBusy(true);
    setAssessError(null);
    try {
      setAssessment(await engine.neoAssess(elements));
    } catch (err) {
      setAssessError(String(err));
    } finally {
      setAssessBusy(false);
    }
  }

  async function runCloseApproach() {
    setApproachBusy(true);
    setApproachError(null);
    try {
      setCloseApproach(await engine.neoCloseApproach(elements, Number(startMjd) || 60000,
                                                      Number(endMjd) || 60400));
    } catch (err) {
      setApproachError(String(err));
    } finally {
      setApproachBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Orbit}
        title="Hazard assessment"
        description="MOID, Tisserand parameter, and PHA classification for a set of orbital elements."
        actions={
          <div className="flex items-end gap-2">
            <ElementFields elements={elements} onChange={setElements} />
            <Button icon={Orbit} disabled={assessBusy} onClick={() => void runAssessment()}>
              {assessBusy ? "Assessing…" : "Assess"}
            </Button>
          </div>
        }
      >
        {assessError && <Note tone="bad">{assessError}</Note>}
        {!assessment && !assessError && (
          <Empty>No assessment run yet. Set orbital elements and assess.</Empty>
        )}
        {assessment && <AssessmentDetail result={assessment} />}
      </Panel>

      <Panel
        icon={Radar}
        title="Close approach"
        description="Minimum geocentric distance over a window, from unperturbed two-body propagation."
        actions={
          <div className="flex items-end gap-2">
            <Field label="Start (MJD)" value={startMjd} onChange={setStartMjd} width="w-24" />
            <Field label="End (MJD)" value={endMjd} onChange={setEndMjd} width="w-24" />
            <Button icon={Radar} disabled={approachBusy} onClick={() => void runCloseApproach()}>
              {approachBusy ? "Computing…" : "Compute"}
            </Button>
          </div>
        }
      >
        {approachError && <Note tone="bad">{approachError}</Note>}
        {!closeApproach && !approachError && (
          <Empty>No close-approach search run yet. Set a window and compute.</Empty>
        )}
        {closeApproach && <CloseApproachDetail result={closeApproach} />}
      </Panel>
    </div>
  );
}

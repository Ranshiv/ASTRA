/** Solar-like oscillation scaling relations (roadmap: astrophysics &
 *  extraterrestrial-study feature pass). Two independent actions:
 *  measure numax/Dnu directly from a stored light curve, or solve the
 *  Kjeldsen & Bedding (1995) scaling relations from manually entered
 *  seismic parameters.
 *
 *  `asteroseismology.measure`/`asteroseismology.solve` are read-only
 *  diagnostic RPC methods, the same category `physical.characterize`/
 *  `digital_twin.*` occupy -- neither is folded into candidate ranking.
 */
import { useState } from "react";
import { AudioWaveform, Calculator } from "lucide-react";

import {
  engine,
  type AsteroseismologyMeasurement,
  type SeismicSolution,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, num } from "@/components/ui";

function SolutionDetail({ solution }: { solution: SeismicSolution }) {
  return (
    <KeyValue
      rows={[
        ["Radius (R_sun)", num(solution.radius_rsun, 3)],
        ["Mass (M_sun)", num(solution.mass_msun, 3)],
        ["log g (cgs)", num(solution.logg_cgs, 3)],
        ["Density (rho_sun)", num(solution.density_rhosun, 3)],
      ]}
    />
  );
}

function MeasurementDetail({ result }: { result: AsteroseismologyMeasurement }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result.quality === "usable" ? "ok" : "bad"}>{result.quality}</Badge>
      </div>
      <KeyValue
        rows={[
          ["numax (uHz)", num(result.numax_uhz, 2)],
          ["Delta nu (uHz)", num(result.delta_nu_uhz, 3)],
        ]}
      />
      {result.solution && <SolutionDetail solution={result.solution} />}
      {result.warnings.length > 0 && (
        <div className="flex flex-col gap-1">
          {result.warnings.map((w) => (
            <Note key={w} tone="muted">{w}</Note>
          ))}
        </div>
      )}
      <Note tone="warn">
        Direct-method scaling-relation values only -- no peakbagging, no individual mode
        frequencies, no red-giant evolutionary classification. Uncorrected for the known
        few-percent Dnu-scaling bias in evolved or metal-poor stars.
      </Note>
    </div>
  );
}

export function AsteroseismologyPanel({ projectId: _projectId }: { projectId?: string }) {
  const [curvePath, setCurvePath] = useState("");
  const [teffK, setTeffK] = useState("5777");
  const [measurement, setMeasurement] = useState<AsteroseismologyMeasurement | null>(null);
  const [measureBusy, setMeasureBusy] = useState(false);
  const [measureError, setMeasureError] = useState<string | null>(null);

  const [numaxUhz, setNumaxUhz] = useState("3090");
  const [deltaNuUhz, setDeltaNuUhz] = useState("135.1");
  const [solveTeffK, setSolveTeffK] = useState("5777");
  const [solution, setSolution] = useState<SeismicSolution | null>(null);
  const [solveBusy, setSolveBusy] = useState(false);
  const [solveError, setSolveError] = useState<string | null>(null);

  async function measureCurve() {
    setMeasureBusy(true);
    setMeasureError(null);
    try {
      setMeasurement(await engine.asteroseismologyMeasure(curvePath, Number(teffK) || undefined));
    } catch (err) {
      setMeasureError(String(err));
    } finally {
      setMeasureBusy(false);
    }
  }

  async function solveRelations() {
    setSolveBusy(true);
    setSolveError(null);
    try {
      setSolution(await engine.asteroseismologySolve(
        Number(numaxUhz) || 0, Number(deltaNuUhz) || 0, Number(solveTeffK) || 0));
    } catch (err) {
      setSolveError(String(err));
    } finally {
      setSolveBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={AudioWaveform}
        title="Measure from a light curve"
        description="Lomb-Scargle power spectrum, numax/Dnu detection, and (with Teff) the scaling-relation solution."
        actions={
          <div className="flex items-end gap-2">
            <Field label="Curve path" value={curvePath} onChange={setCurvePath} width="w-56" />
            <Field label="Teff (K)" value={teffK} onChange={setTeffK} width="w-20" />
            <Button icon={AudioWaveform} disabled={measureBusy || !curvePath}
                    onClick={() => void measureCurve()}>
              {measureBusy ? "Measuring…" : "Measure"}
            </Button>
          </div>
        }
      >
        {measureError && <Note tone="bad">{measureError}</Note>}
        {!measurement && !measureError && (
          <Empty>No measurement run yet. Enter a stored curve path and measure.</Empty>
        )}
        {measurement && <MeasurementDetail result={measurement} />}
      </Panel>

      <Panel
        icon={Calculator}
        title="Solve scaling relations"
        description="Kjeldsen & Bedding (1995) direct-method inversion from (numax, Delta nu, Teff)."
        actions={
          <div className="flex items-end gap-2">
            <Field label="numax (uHz)" value={numaxUhz} onChange={setNumaxUhz} width="w-24" />
            <Field label="Delta nu (uHz)" value={deltaNuUhz} onChange={setDeltaNuUhz} width="w-24" />
            <Field label="Teff (K)" value={solveTeffK} onChange={setSolveTeffK} width="w-20" />
            <Button icon={Calculator} disabled={solveBusy} onClick={() => void solveRelations()}>
              {solveBusy ? "Solving…" : "Solve"}
            </Button>
          </div>
        }
      >
        {solveError && <Note tone="bad">{solveError}</Note>}
        {!solution && !solveError && (
          <Empty>No solution computed yet. Enter seismic parameters and solve.</Empty>
        )}
        {solution && <SolutionDetail solution={solution} />}
      </Panel>
    </div>
  );
}

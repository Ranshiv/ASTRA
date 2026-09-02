/** Transmission-spectrum band-detection significance (roadmap:
 *  astrophysics & extraterrestrial-study feature pass).
 *
 *  `biosignature.synthesize`/`fit`/`detect` are read-only diagnostic RPC
 *  methods, the same category `physical.characterize`/`digital_twin.*`
 *  occupy. This panel demonstrates the model on a SYNTHETIC spectrum
 *  (no live JWST spectrum ingestion path exists -- see `biosignature.py`'s
 *  `[GAP]`), and it must never let a fitted band amplitude read as a
 *  molecular abundance: the caveat is a persistent, un-dismissable Note,
 *  not a footnote.
 */
import { useState } from "react";
import { Beaker, Waves } from "lucide-react";

import {
  engine,
  type BiosignatureDetectResult,
  type BiosignatureSynthesis,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Select, Table, num } from "@/components/ui";

const MOLECULE_OPTIONS = ["H2O", "CH4", "CO2", "CO", "O2", "O3"].map((m) => ({ value: m, label: m }));

function DetectionTable({ result }: { result: BiosignatureDetectResult }) {
  const rows = Object.values(result.significances);
  return (
    <div className="flex flex-col gap-3">
      <Table head={["Molecule", "Delta BIC", "Detected"]}>
        {rows.map((row) => (
          <tr key={row.molecule} className="border-b border-[var(--color-edge)]/50">
            <td className="px-2 py-1.5">{row.molecule}</td>
            <td className="px-2 py-1.5">{num(row.delta_bic, 2)}</td>
            <td className="px-2 py-1.5">
              <Badge tone={row.detected ? "warn" : "muted"}>{row.detected ? "detected" : "not detected"}</Badge>
            </td>
          </tr>
        ))}
      </Table>
      {result.disequilibrium.co_detection_flag && (
        <Badge tone="bad">CH4 + oxidant co-detection screening flag</Badge>
      )}
      <Note tone="warn">{result.disequilibrium.caveat}</Note>
      <Note tone="bad">
        A "detected" band is a statistical band-detection significance (Delta BIC &gt; 10), NOT a
        calibrated molecular abundance. This engine has no line-by-line opacity model; no
        percentage or ppm abundance can be honestly reported here.
      </Note>
    </div>
  );
}

export function BiosignaturePanel({ projectId: _projectId }: { projectId?: string }) {
  const [molecule, setMolecule] = useState("H2O");
  const [logAmplitude, setLogAmplitude] = useState("0.5");
  const [errorPpm, setErrorPpm] = useState("50");

  const [spectrum, setSpectrum] = useState<BiosignatureSynthesis | null>(null);
  const [synthBusy, setSynthBusy] = useState(false);
  const [synthError, setSynthError] = useState<string | null>(null);

  const [detection, setDetection] = useState<BiosignatureDetectResult | null>(null);
  const [detectBusy, setDetectBusy] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);

  async function synthesize() {
    setSynthBusy(true);
    setSynthError(null);
    setDetection(null);
    try {
      setSpectrum(await engine.biosignatureSynthesize({
        stellarRadiusRsun: 1.0, planetMassMjup: 1.0, temperatureK: 1000.0,
        referenceRadiusRjup: 1.0, abundances: [[molecule, Number(logAmplitude) || 0]],
        crossSections: { [molecule]: 5.0 }, nPoints: 40,
        errorPpm: Number(errorPpm) || 50,
      }));
    } catch (err) {
      setSynthError(String(err));
    } finally {
      setSynthBusy(false);
    }
  }

  async function detect() {
    if (!spectrum) return;
    setDetectBusy(true);
    setDetectError(null);
    try {
      setDetection(await engine.biosignatureDetect(
        spectrum.wavelength_um, spectrum.depth, spectrum.error, 1.0, 1.0,
        MOLECULE_OPTIONS.map((m) => m.value), { [molecule]: 5.0 }));
    } catch (err) {
      setDetectError(String(err));
    } finally {
      setDetectBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Waves}
        title="Synthetic transmission spectrum"
        description="Generate a demo spectrum from the isothermal forward model -- no live JWST spectrum ingestion path exists."
        actions={
          <div className="flex items-end gap-2">
            <Select label="Molecule" value={molecule} options={MOLECULE_OPTIONS} onChange={setMolecule} />
            <Field label="log10 amplitude" value={logAmplitude} onChange={setLogAmplitude} width="w-24" />
            <Field label="Error (ppm)" value={errorPpm} onChange={setErrorPpm} width="w-20" />
            <Button icon={Waves} disabled={synthBusy} onClick={() => void synthesize()}>
              {synthBusy ? "Generating…" : "Generate"}
            </Button>
          </div>
        }
      >
        {synthError && <Note tone="bad">{synthError}</Note>}
        {!spectrum && !synthError && (
          <Empty>No spectrum generated yet. Pick a molecule and amplitude and generate one.</Empty>
        )}
        {spectrum && (
          <KeyValue
            rows={[
              ["Points", spectrum.wavelength_um.length],
              ["Wavelength range (um)", `${num(spectrum.wavelength_um[0], 2)} - ${num(spectrum.wavelength_um[spectrum.wavelength_um.length - 1], 2)}`],
              ["Depth range", `${num(Math.min(...spectrum.depth), 6)} - ${num(Math.max(...spectrum.depth), 6)}`],
            ]}
          />
        )}
      </Panel>

      <Panel
        icon={Beaker}
        title="Band-detection significance"
        description="Delta BIC between the full model and a flat-line null, for each molecule."
        actions={
          <Button icon={Beaker} disabled={detectBusy || !spectrum} onClick={() => void detect()}>
            {detectBusy ? "Fitting…" : "Detect bands"}
          </Button>
        }
      >
        {detectError && <Note tone="bad">{detectError}</Note>}
        {!detection && !detectError && (
          <Empty>No detection run yet. Generate a spectrum first, then detect bands.</Empty>
        )}
        {detection && <DetectionTable result={detection} />}
      </Panel>
    </div>
  );
}

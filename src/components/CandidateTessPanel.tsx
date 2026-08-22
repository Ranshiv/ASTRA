import { ChevronRight, Satellite } from "lucide-react";
import { useState } from "react";

import { LightCurvePlot } from "@/components/LightCurvePlot";
import { Button, Field, Note } from "@/components/ui";
import { engine, type Candidate, type CurvePayload, type TessPhotometryPayload, type TessTpfProduct } from "@/lib/engine";

function asTessCurve(payload: TessPhotometryPayload): CurvePayload {
  const finite = payload.flux.filter(Number.isFinite);
  const mean = finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
  const variance = finite.length
    ? finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length
    : 0;
  return {
    path: payload.curve_path ?? payload.path,
    survey: "TESS",
    release: `tpf-s${payload.sector ?? "unknown"}`,
    object_id: payload.source ?? "TPF",
    band: "TESS",
    value_kind: "flux",
    time_system: "BJD_TDB",
    points: payload.points,
    time_span_days: payload.time.length > 1 ? payload.time[payload.time.length - 1] - payload.time[0] : 0,
    mean_value: mean,
    std_value: Math.sqrt(variance),
    time: payload.time,
    value: payload.flux,
    value_err: payload.flux_err.map((value) => value ?? Number.NaN),
    shown_points: payload.shown_points,
    downsampled: payload.downsampled,
  };
}

export function CandidateTessPanel({ candidate }: { candidate: Candidate }) {
  const [tessSector, setTessSector] = useState("1");
  const [tessProduct, setTessProduct] = useState<TessTpfProduct | null>(null);
  const [tessPhotometry, setTessPhotometry] = useState<TessPhotometryPayload | null>(null);
  const [tessStatus, setTessStatus] = useState("");

  async function downloadTessTpf() {
    const sector = Number(tessSector);
    if (!Number.isInteger(sector) || sector < 1) {
      setTessStatus("Enter a valid TESS sector number.");
      return;
    }
    setTessStatus("Queueing an explicit, quota-safe TESS target-pixel download…");
    try {
      const job = await engine.tessTpfDownload({
        raDeg: candidate.ra_deg,
        decDeg: candidate.dec_deg,
        sector,
        targetId: candidate.object_id || candidate.candidate_id,
      });
      let status = await engine.jobStatus(job.job_id);
      while (["queued", "running", "retrying"].includes(status.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 750));
        status = await engine.jobStatus(job.job_id);
      }
      if (status.status !== "completed") throw new Error(status.error ?? `TPF download ${status.status}`);
      const product = status.result as TessTpfProduct;
      setTessProduct(product);
      setTessStatus(
        product.reused
          ? "Using the existing verified TESS target-pixel file."
          : "TESS target-pixel file downloaded, checksummed, and validated.",
      );
    } catch (err) {
      setTessStatus(String(err));
    }
  }

  async function extractTessPhotometry() {
    if (!tessProduct) return;
    setTessStatus("Extracting background-subtracted aperture photometry…");
    try {
      const payload = await engine.tessTpfPhotometry({
        path: tessProduct.path,
        raDeg: candidate.ra_deg,
        decDeg: candidate.dec_deg,
        targetId: candidate.object_id || candidate.candidate_id,
        persist: true,
      });
      setTessPhotometry(payload);
      setTessStatus(`${payload.points.toLocaleString()} quality-filtered cadences persisted. Blend risk: ${payload.blend.risk}.`);
    } catch (err) {
      setTessStatus(String(err));
    }
  }

  return (
    <details className="group rounded border border-[var(--color-edge)] p-2">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs [&::-webkit-details-marker]:hidden">
        <ChevronRight
          size={12}
          strokeWidth={2}
          className="text-[var(--color-muted)] transition-transform duration-200 group-open:rotate-90"
        />
        <Satellite size={13} strokeWidth={2} />
        TESS target-pixel file
      </summary>
      <p className="mt-1 text-xs text-[var(--color-muted)]">
        Candidate-scale only. TESS pixels span about 21 arcsec, so this remains neighbourhood photometry rather
        than resolved stellar confirmation.
      </p>
      <div className="my-2 flex flex-wrap items-end gap-1.5">
        <Field id="tess-sector" label="TESS sector" value={tessSector} onChange={setTessSector} width="w-20" />
        <Button onClick={() => void downloadTessTpf()}>Download TPF</Button>
        {tessProduct && (
          <Button onClick={() => void extractTessPhotometry()} tone="accent">
            Extract aperture curve
          </Button>
        )}
      </div>
      {tessStatus && <Note>{tessStatus}</Note>}
      {tessProduct && <p className="mt-1 break-all font-mono text-xs text-[var(--color-accent)]">{tessProduct.path}</p>}
      {tessPhotometry && (
        <>
          <LightCurvePlot curve={asTessCurve(tessPhotometry)} folded={null} />
          <div className="mt-1">
            <Note tone="warn">
              Blend assessment: {tessPhotometry.blend.risk} risk; {tessPhotometry.blend.neighbors_in_aperture}{" "}
              neighbour(s) in the aperture. {tessPhotometry.blend.warning}
            </Note>
          </div>
        </>
      )}
    </details>
  );
}

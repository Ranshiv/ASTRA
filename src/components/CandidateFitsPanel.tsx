import { ChevronRight, Image as ImageIcon } from "lucide-react";
import { useState } from "react";

import { FitsViewer } from "@/components/FitsViewer";
import { Button, Field, Note } from "@/components/ui";
import {
  engine,
  type Candidate,
  type ImageFeaturePayload,
  type ImageProduct,
  type ZtfImageMetadata,
} from "@/lib/engine";

export function CandidateFitsPanel({ candidate, projectId }: { candidate: Candidate; projectId?: string }) {
  const [fitsPath, setFitsPath] = useState("");
  const [cutouts, setCutouts] = useState<ZtfImageMetadata[]>([]);
  const [products, setProducts] = useState<ImageProduct[]>([]);
  const [cutoutStatus, setCutoutStatus] = useState("");
  const [imageFeatures, setImageFeatures] = useState<ImageFeaturePayload | null>(null);

  async function findCutouts() {
    setCutoutStatus("Searching ZTF image metadata…");
    try {
      const found = await engine.ztfImageSearch(candidate.ra_deg, candidate.dec_deg);
      setCutouts(found);
      setCutoutStatus(
        found.length ? `${found.length} archive image products found.` : "No overlapping ZTF image products found.",
      );
    } catch (err) {
      setCutoutStatus(String(err));
    }
  }

  async function downloadCutout(metadata: ZtfImageMetadata) {
    setCutoutStatus("Queueing a quota-safe FITS cutout download…");
    try {
      const job = await engine.ztfImageDownload({ raDeg: candidate.ra_deg, decDeg: candidate.dec_deg, metadata });
      let status = await engine.jobStatus(job.job_id);
      while (["queued", "running", "retrying"].includes(status.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 750));
        status = await engine.jobStatus(job.job_id);
      }
      if (status.status !== "completed") throw new Error(status.error ?? `download ${status.status}`);
      const product = status.result as ImageProduct;
      setProducts((current) => [product, ...current.filter((item) => item.product_id !== product.product_id)]);
      setFitsPath(product.path);
      setCutoutStatus(
        product.reused ? "Using the existing verified FITS cutout." : "FITS cutout downloaded, checksummed, and validated.",
      );
    } catch (err) {
      setCutoutStatus(String(err));
    }
  }

  async function extractFeatures() {
    try {
      setCutoutStatus("Extracting bounded image morphology features…");
      const result = await engine.imageFeatures(fitsPath, projectId);
      setImageFeatures(result);
      setCutoutStatus(result.output_path ? `Image features persisted: ${result.output_path}` : "Image features extracted.");
    } catch (err) {
      setCutoutStatus(String(err));
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
        <ImageIcon size={13} strokeWidth={2} />
        FITS image
      </summary>
      <div className="my-2 flex flex-wrap gap-1.5">
        <Button onClick={() => void findCutouts()}>Find ZTF cutouts</Button>
        {cutouts.slice(0, 5).map((item, index) => (
          <Button key={`${item.filefracday}-${index}`} onClick={() => void downloadCutout(item)}>
            Download {String(item.filtercode)}
          </Button>
        ))}
      </div>
      {cutoutStatus && <Note>{cutoutStatus}</Note>}
      {products.map((item) => (
        <button
          type="button"
          key={item.product_id}
          onClick={() => setFitsPath(item.path)}
          className="block max-w-full truncate text-left font-mono text-xs text-[var(--color-accent)]"
        >
          {item.product_id.slice(0, 12)} · {(item.bytes / 1024).toFixed(0)} KiB
        </button>
      ))}
      <div className="my-2">
        <Field
          id="fits-path"
          label="Path to a local FITS cutout"
          value={fitsPath}
          onChange={setFitsPath}
          placeholder="Path to a local FITS cutout"
          width="w-full"
        />
      </div>
      {fitsPath ? (
        <>
          <FitsViewer path={fitsPath} />
          <div className="mt-2">
            <Button onClick={() => void extractFeatures()}>Extract image features</Button>
          </div>
          {imageFeatures && (
            <div className="mt-2 rounded border border-[var(--color-edge)] bg-[var(--color-panel-2)]/30 p-2">
              <p className="text-xs font-medium">Image-derived features</p>
              <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                {Object.entries(imageFeatures.features).map(([key, value]) => (
                  <>
                    <dt key={`${key}-label`} className="text-[var(--color-muted)]">
                      {key.replace(/_/g, " ")}
                    </dt>
                    <dd key={`${key}-value`} className="font-mono">
                      {value === null ? "—" : typeof value === "number" ? value.toFixed(4) : String(value)}
                    </dd>
                  </>
                ))}
              </dl>
            </div>
          )}
        </>
      ) : null}
    </details>
  );
}

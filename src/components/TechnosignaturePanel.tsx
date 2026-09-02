/** Narrowband drift (de-Doppler) technosignature search, demonstrated on
 *  a SYNTHETIC dynamic spectrum (roadmap: astrophysics & extraterrestrial-
 *  study feature pass). There is no filterbank reader and no live
 *  Breakthrough Listen data path in this engine -- see
 *  `technosignature.py`'s module docstring `[GAP]`. Every number this
 *  panel shows is measured on injected synthetic data; it is a
 *  demonstration of the search algorithm, not a search of real
 *  observations.
 *
 *  `technosignature.search` is a read-only diagnostic RPC method, the
 *  same category `physical.characterize`/`digital_twin.*` occupy.
 */
import { useState } from "react";
import { Radio } from "lucide-react";

import { engine, type TechnosignatureSearchResult } from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Table, num } from "@/components/ui";

function SearchResultDetail({ result }: { result: TechnosignatureSearchResult }) {
  const topHits = [...result.hits].sort((a, b) => b.snr - a.snr).slice(0, 10);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result.hits.length > 0 ? "warn" : "muted"}>
          {result.hits.length} hit{result.hits.length === 1 ? "" : "s"} above SNR {num(result.snr_threshold, 0)}
        </Badge>
        <Badge tone="muted">{result.n_drift_trials} drift trials, +/- {num(result.max_drift_hz_s, 1)} Hz/s</Badge>
      </div>
      {topHits.length > 0 && (
        <Table head={["Channel", "Drift (Hz/s)", "SNR"]}>
          {topHits.map((hit, i) => (
            <tr key={i} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{hit.freq_channel_index}</td>
              <td className="px-2 py-1.5">{num(hit.drift_rate_hz_s, 3)}</td>
              <td className="px-2 py-1.5">{num(hit.snr, 1)}</td>
            </tr>
          ))}
        </Table>
      )}
      <Note tone="warn">
        Measured entirely on synthetic injected data -- this engine has no real Breakthrough
        Listen data path. A surviving hit is an unexplained narrowband detection in the
        synthetic array, nothing more; no ON/OFF cadence rejection has been applied here.
      </Note>
    </div>
  );
}

export function TechnosignaturePanel({ projectId: _projectId }: { projectId?: string }) {
  const [driftRate, setDriftRate] = useState("2.0");
  const [snr, setSnr] = useState("50");
  const [result, setResult] = useState<TechnosignatureSearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch() {
    setBusy(true);
    setError(null);
    try {
      setResult(await engine.technosignatureSearch({
        driftRateHzS: Number(driftRate) || 0, snr: Number(snr) || 0,
      }));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Radio}
        title="Synthetic drift search"
        description="Inject a drifting narrowband tone into synthetic detector noise and run the de-Doppler search."
        actions={
          <div className="flex items-end gap-2">
            <Field label="Drift rate (Hz/s)" value={driftRate} onChange={setDriftRate} width="w-28" />
            <Field label="Injected SNR" value={snr} onChange={setSnr} width="w-24" />
            <Button icon={Radio} disabled={busy} onClick={() => void runSearch()}>
              {busy ? "Searching…" : "Search"}
            </Button>
          </div>
        }
      >
        {error && <Note tone="bad">{error}</Note>}
        {!result && !error && (
          <Empty>No search run yet. Set an injected drift rate and SNR, then search.</Empty>
        )}
        {result && (
          <div className="flex flex-col gap-2">
            <KeyValue
              rows={[
                ["Injected drift (Hz/s)", num(result.truth.drift_rate_hz_s, 3)],
                ["Injected SNR", num(result.truth.snr, 1)],
                ["Injected channel", result.truth.start_channel ?? "—"],
              ]}
            />
            <SearchResultDetail result={result} />
          </div>
        )}
      </Panel>
    </div>
  );
}

/** Read-only diagnostics for the survey digital twin (backlog item 42):
 *  fit a per-survey cadence/noise profile from real stored curves, sample a
 *  synthetic batch from it, and report the two success criteria the
 *  research module itself defines -- distance between simulated and real
 *  summary statistics, and transfer performance.
 *
 *  Deliberately never touches ranking: every value here comes back through
 *  `digital_twin.*`, a read-only diagnostic RPC group, the same category
 *  `physical.characterize`/`significance.calibrate` already occupy (see
 *  each `rpc.py` handler's own docstring). This panel only displays what
 *  they return; it never writes anything back into a candidate's score.
 */
import { useState } from "react";
import { FlaskConical, GitCompare, Sparkles, Waves } from "lucide-react";

import {
  engine,
  type DigitalTwinDistance,
  type DigitalTwinSample,
  type DigitalTwinTransferResult,
  type SurveyProfileSummary,
} from "@/lib/engine";
import { Badge, Button, Empty, KeyValue, Note, Panel, Select, Table, num } from "@/components/ui";

const SURVEY_OPTIONS = [
  { value: "ZTF", label: "ZTF" },
  { value: "TESS", label: "TESS" },
];

function ProfileSummary({ profile }: { profile: SurveyProfileSummary }) {
  if (profile.note) {
    return <Note tone="warn">{profile.note}</Note>;
  }
  return (
    <KeyValue
      rows={[
        ["Curves used", profile.n_curves_used],
        ["Mean coverage", num(profile.mean_coverage, 4)],
        ["Gap runs sampled", profile.n_gap_runs_sampled],
        ["Noise std", num(profile.noise_std, 4)],
        ["Sequence length", profile.length],
      ]}
    />
  );
}

function DistancePanel({ result }: { result: DigitalTwinDistance }) {
  if (result.error) {
    return <Note tone="warn">{result.error}</Note>;
  }
  if (result.note) {
    return <Note tone="muted">{result.note}</Note>;
  }
  const rows = Object.entries(result.per_feature);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result.mean_ks_statistic < 0.2 ? "ok" : result.mean_ks_statistic < 0.5 ? "warn" : "bad"}>
          mean KS {num(result.mean_ks_statistic, 3)}
        </Badge>
        <Badge tone="muted">{result.real_rows} real vs {result.synthetic_rows} synthetic</Badge>
      </div>
      <Table head={["Feature", "KS distance"]}>
        {rows.map(([name, value]) => (
          <tr key={name} className="border-b border-[var(--color-edge)]/50">
            <td className="px-2 py-1.5">{name.replace(/_/g, " ")}</td>
            <td className="px-2 py-1.5">{value === null ? "—" : num(value, 4)}</td>
          </tr>
        ))}
      </Table>
      <Note tone="warn">
        0 means indistinguishable, 1 means no shared support. This does not rank or
        score anything -- it only measures how close the digital twin is to real data.
      </Note>
    </div>
  );
}

function TransferPanel({ result }: { result: DigitalTwinTransferResult }) {
  if (result.error) {
    return <Note tone="warn">{result.error}</Note>;
  }
  return (
    <div className="flex flex-col gap-2">
      <Table head={["Arm", "Mean ROC-AUC", "Std", "CI95", "Seeds"]}>
        {(["trained_on_real", "trained_on_synthetic"] as const).map((arm) => {
          const value = result[arm];
          return (
            <tr key={arm} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{arm === "trained_on_real" ? "trained on real" : "trained on synthetic"}</td>
              <td className="px-2 py-1.5">{value ? num(value.mean, 4) : "—"}</td>
              <td className="px-2 py-1.5">{value ? num(value.std, 4) : "—"}</td>
              <td className="px-2 py-1.5">
                {value ? `[${num(value.ci95[0], 3)}, ${num(value.ci95[1], 3)}]` : "—"}
              </td>
              <td className="px-2 py-1.5">{value?.n ?? "—"}</td>
            </tr>
          );
        })}
      </Table>
      <Note tone="warn">
        Neither arm is declared the winner here -- overlapping confidence intervals mean
        the two are not reliably distinguishable at this sample size. Both are trained on
        the identical architecture and scored on the same held-out real data.
      </Note>
    </div>
  );
}

export function DigitalTwinPanel({ projectId: _projectId }: { projectId?: string }) {
  const [survey, setSurvey] = useState("ZTF");
  const [limit, setLimit] = useState("500");

  const [sample, setSample] = useState<DigitalTwinSample | null>(null);
  const [sampleBusy, setSampleBusy] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);

  const [distance, setDistance] = useState<DigitalTwinDistance | null>(null);
  const [distanceBusy, setDistanceBusy] = useState(false);
  const [distanceError, setDistanceError] = useState<string | null>(null);

  const [transfer, setTransfer] = useState<DigitalTwinTransferResult | null>(null);
  const [transferBusy, setTransferBusy] = useState(false);
  const [transferError, setTransferError] = useState<string | null>(null);

  const parsedLimit = Number(limit) || 500;

  async function fitAndSample() {
    setSampleBusy(true);
    setSampleError(null);
    try {
      setSample(await engine.digitalTwinSample(survey, parsedLimit));
    } catch (err) {
      setSampleError(String(err));
    } finally {
      setSampleBusy(false);
    }
  }

  async function evaluateDistance() {
    setDistanceBusy(true);
    setDistanceError(null);
    try {
      setDistance(await engine.digitalTwinEvaluateDistance(survey, parsedLimit));
    } catch (err) {
      setDistanceError(String(err));
    } finally {
      setDistanceBusy(false);
    }
  }

  async function evaluateTransfer() {
    setTransferBusy(true);
    setTransferError(null);
    try {
      setTransfer(await engine.digitalTwinEvaluateTransfer(survey, parsedLimit));
    } catch (err) {
      setTransferError(String(err));
    } finally {
      setTransferBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Sparkles}
        title="Survey digital twin"
        description="Fit a cadence/noise profile from real stored curves, then sample a synthetic population from it."
        actions={
          <div className="flex items-end gap-2">
            <Select label="Survey" value={survey} options={SURVEY_OPTIONS} onChange={setSurvey} />
            <input
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
              aria-label="Curve limit"
              className="h-8 w-20 rounded border border-[var(--color-edge)] bg-[var(--color-panel-2)] px-2 text-xs"
            />
            <Button icon={Waves} disabled={sampleBusy} onClick={() => void fitAndSample()}>
              {sampleBusy ? "Fitting…" : "Fit & sample"}
            </Button>
          </div>
        }
      >
        {sampleError && <Note tone="bad">{sampleError}</Note>}
        {!sample && !sampleError && (
          <Empty>No profile fitted yet. Pick a survey and fit one from real stored curves.</Empty>
        )}
        {sample && (
          <div className="flex flex-col gap-3">
            <ProfileSummary profile={sample.profile} />
            <KeyValue
              rows={[
                ["Synthetic rows sampled", sample.batch.rows],
                ["Synthetic mean coverage", num(sample.batch.mean_coverage, 4)],
              ]}
            />
          </div>
        )}
      </Panel>

      <Panel
        icon={GitCompare}
        title="Distance to real data"
        description="Per-feature Kolmogorov-Smirnov distance between the synthetic and real populations."
        actions={
          <Button icon={GitCompare} disabled={distanceBusy} onClick={() => void evaluateDistance()}>
            {distanceBusy ? "Measuring…" : "Measure distance"}
          </Button>
        }
      >
        {distanceError && <Note tone="bad">{distanceError}</Note>}
        {!distance && !distanceError && (
          <Empty>No distance measurement run yet for this survey.</Empty>
        )}
        {distance && <DistancePanel result={distance} />}
      </Panel>

      <Panel
        icon={FlaskConical}
        title="Transfer performance"
        description="Train the same detector on real vs. synthetic data; score both on the same held-out real set. Minutes, not seconds."
        actions={
          <Button icon={FlaskConical} disabled={transferBusy} onClick={() => void evaluateTransfer()}>
            {transferBusy ? "Training…" : "Run transfer study"}
          </Button>
        }
      >
        {transferError && <Note tone="bad">{transferError}</Note>}
        {!transfer && !transferError && (
          <Empty>No transfer study run yet. This trains real models and can take a few minutes.</Empty>
        )}
        {transfer && <TransferPanel result={transfer} />}
      </Panel>
    </div>
  );
}

import { RefreshCcw, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

import { LightCurvePlot } from "@/components/LightCurvePlot";
import { Button, Field, Note } from "@/components/ui";
import { engine, type CurvePayload, type FoldedCurve } from "@/lib/engine";

export function CandidateFoldPanel({
  path,
  bestPeriodDays,
}: {
  path: string;
  bestPeriodDays?: number | null;
}) {
  const [curve, setCurve] = useState<CurvePayload | null>(null);
  const [folded, setFolded] = useState<FoldedCurve | null>(null);
  const [period, setPeriod] = useState("");
  const [foldError, setFoldError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCurve(null);
    setFolded(null);
    setFoldError(null);
    setPeriod(
      typeof bestPeriodDays === "number" && Number.isFinite(bestPeriodDays) && bestPeriodDays > 0
        ? String(bestPeriodDays)
        : "",
    );
    if (path) {
      engine
        .curveGet(path, 2000, "BJD_TDB")
        .then((result) => {
          if (!cancelled) setCurve(result);
        })
        .catch(() => {
          /* detail remains usable */
        });
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  async function applyFold() {
    if (!curve) return;
    const days = Number(period);
    if (!Number.isFinite(days) || days <= 0) {
      setFoldError("Enter a positive period in days.");
      return;
    }
    setFoldError(null);
    try {
      setFolded(await engine.curveFold(curve.path, days));
    } catch (err) {
      setFoldError(String(err));
    }
  }

  if (!curve) return null;

  return (
    <div>
      <LightCurvePlot curve={curve} folded={folded} />
      <div className="mt-1 flex items-end gap-2">
        <Field
          id="candidate-fold-period"
          label="Fold period (days)"
          value={period}
          onChange={setPeriod}
          placeholder="e.g. 3.14"
          width="flex-1"
        />
        <Button onClick={applyFold} icon={RefreshCcw}>
          Fold
        </Button>
        {folded && (
          <Button onClick={() => setFolded(null)} icon={RotateCcw}>
            Unfold
          </Button>
        )}
      </div>
      {foldError && (
        <div className="mt-1">
          <Note tone="bad">{foldError}</Note>
        </div>
      )}
    </div>
  );
}

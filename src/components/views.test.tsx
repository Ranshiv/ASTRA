/** Empty and refusal states for the plan section 10 views.
 *
 * A fresh installation has no projects, no candidates and no experiments, and a
 * released build has no PyTorch. Those are the states a new user sees first, so
 * they are the ones worth pinning: each must explain itself rather than render
 * an empty table that looks like a failure.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { invoke } from "@/test/setup";
import { CrossSurveyPanel } from "@/components/CrossSurveyPanel";
import { ExperimentsView } from "@/components/ExperimentsView";
import { ModelsView } from "@/components/ModelsView";
import { ReportsView } from "@/components/ReportsView";

const DEEP_UNAVAILABLE =
  "PyTorch is not available in this build, so deep models cannot run. " +
  "Released ASTRA installers ship a CPU-only engine.";

/** Answer each command with whatever the view expects for an empty engine. */
function respond(overrides: Record<string, unknown> = {}) {
  const defaults: Record<string, unknown> = {
    engine_experiments: [],
    engine_features_list: [],
    engine_ranker_list: [],
    engine_versions: { numpy: "2.5.2", torch: "not installed" },
    engine_manifests: [],
    engine_label_summary: { total: 0, by_label: { interesting: 0, artifact: 0 } },
    engine_candidates: { count: 0, candidates: [] },
    engine_crossmatch: {
      summary: { groups: 0, multi_survey: 0, ambiguous: 0, resolved_multi_survey: 0, by_survey_count: {} },
      groups: [],
    },
    engine_profiles: { profiled: 0, profiles: [] },
  };
  const table = { ...defaults, ...overrides };
  invoke.mockImplementation((command: string) => {
    if (!(command in table)) return Promise.resolve({});
    const value = table[command];
    return value instanceof Error ? Promise.reject(value) : Promise.resolve(value);
  });
}

describe("ExperimentsView", () => {
  it("explains an empty record set instead of showing a blank table", async () => {
    respond();
    render(<ExperimentsView />);

    expect(await screen.findByText(/No experiments recorded yet/i)).toBeInTheDocument();
  });

  it("warns that pooling ZTF and TESS measures the mixture", async () => {
    respond();
    render(<ExperimentsView />);

    expect(
      await screen.findByText(/pools ZTF and TESS/i),
    ).toBeInTheDocument();
  });

  it("surfaces an engine error rather than an empty list", async () => {
    respond({ engine_experiments: new Error("engine not running") });
    render(<ExperimentsView />);

    expect(await screen.findByText(/engine not running/i)).toBeInTheDocument();
  });
});

describe("ModelsView", () => {
  it("reports an empty feature-matrix list", async () => {
    respond();
    render(<ModelsView />);

    expect(await screen.findByText(/No feature matrices built yet/i)).toBeInTheDocument();
  });

  it("names the trade-off when the build has no PyTorch", async () => {
    respond();
    render(<ModelsView />);

    const button = await screen.findByRole("button", { name: /Train deep model/i });
    invoke.mockRejectedValue(DEEP_UNAVAILABLE);
    fireEvent.click(button);

    // The engine's own explanation, not a bare ModuleNotFoundError.
    expect(await screen.findByText(/CPU-only engine/i)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText(/CPU-only build/i)).toBeInTheDocument(),
    );
  });
});

describe("ReportsView", () => {
  it("says where exports go before any have been written", async () => {
    respond();
    render(<ReportsView />);

    expect(await screen.findByText(/reports directory/i)).toBeInTheDocument();
  });

  it("explains the review gate rather than showing a metric", async () => {
    respond();
    render(<ReportsView />);

    const evaluate = await screen.findByRole("button", { name: /Evaluate/i });
    invoke.mockResolvedValue({
      ready: false, reason: "insufficient independent human labels",
      minimum_labels: 50, minimum_per_class: 10,
      labels: 1, positives: 1, negatives: 0,
    });
    fireEvent.click(evaluate);

    expect(
      await screen.findByText(/look like a result without being one/i),
    ).toBeInTheDocument();
  });
});

describe("CrossSurveyPanel", () => {
  it("distinguishes resolved from merely detected surveys", async () => {
    respond();
    render(<CrossSurveyPanel />);

    expect(
      await screen.findByText(/corroborates the neighbourhood/i),
    ).toBeInTheDocument();
  });

  it("shows the evidence a profile is missing, not just its score", async () => {
    respond({
      engine_profiles: {
        profiled: 1,
        profiles: [{
          independent_surveys: 2, resolved_surveys: 1,
          views: [{
            survey: "ZTF", object_id: "z1", band: "g", value_kind: "mag",
            points: 350, reduced_chi2: 120.0, best_period_days: 0.5668,
            period_snr: 21.0, robust_amplitude: 0.42,
            fractional_amplitude: 0.387, baseline_days: 2740.0,
          }],
          separations_arcsec: { ZTF: 0.0, TESS: 8.0 },
          ambiguous: [], blended: ["TESS"],
          consistency: 0.62, weight_version: 2, weight_used: 0.9,
          period_fap: 0.0755,
          components: { independent_detection: 0.33 },
          notes: ["Blended in TESS."],
        }],
      },
    });
    render(<CrossSurveyPanel />);

    // weight_used below 1.0 means the score used only part of the evidence.
    expect(await screen.findByText(/weight used 0\.90/i)).toBeInTheDocument();
    expect(screen.getByText(/period FAP 7\.5%/i)).toBeInTheDocument();
    expect(screen.getByText(/TESS blended/i)).toBeInTheDocument();
  });
});

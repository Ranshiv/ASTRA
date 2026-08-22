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
import { ExplainPanel, FrbCoincidence, GwCoincidence } from "@/components/ExplainPanel";
import { ExperimentsView } from "@/components/ExperimentsView";
import { SkyExplorer } from "@/components/SkyExplorer";
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
    engine_candidates_spatial: {
      points: [], total: 0, reliable: 0, snr_threshold: 5.0,
      gaia_matched: 0, gaia_match_rate: null,
    },
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

/** One ranked candidate carrying the explanation fields the engine always
 *  computes: weighted contributions, a partial weight, and a fired indicator. */
const EXPLAINED_CANDIDATE = {
  candidate_id: "cand-1",
  rank: 1,
  object_id: "ZTF18abc",
  survey: "ZTF",
  release: "dr24",
  band: "g",
  path: "curves/ztf/z1.parquet",
  ra_deg: 180.122,
  dec_deg: 22.411,
  score: {
    total: 0.6212,
    weight_used: 0.85,
    weight_version: 1,
    components: { statistical_rarity: 0.9, data_quality: 0.8 },
    weighted: { statistical_rarity: 0.225, data_quality: 0.04 },
    reasons: ["Rarest 1% of the population."],
  },
  artifact: {
    likelihood: 0.42,
    verdict: "possibly an artifact",
    indicators: [
      { name: "sampling_period", weight: 0.35, detail: "Period matches the 0.5 d cadence." },
    ],
    clearing_evidence: ["Detected independently in Gaia."],
  },
  features: {},
  explanation: {
    what_happened: "Brightened by 0.4 mag over two seasons.",
    coverage: { tier: "A", status: "periodic features available" },
  },
};

describe("ExplainPanel", () => {
  it("explains an empty candidate set instead of showing a blank table", async () => {
    respond();
    render(<ExplainPanel />);

    expect(await screen.findByText(/No candidates to explain yet/i)).toBeInTheDocument();
  });

  it("surfaces an engine error rather than an empty list", async () => {
    respond({ engine_candidates: new Error("engine not running") });
    render(<ExplainPanel />);

    expect(await screen.findByText(/engine not running/i)).toBeInTheDocument();
  });

  it("shows each component's contribution, not just its raw score", async () => {
    respond({ engine_candidates: { count: 1, candidates: [EXPLAINED_CANDIDATE] } });
    render(<ExplainPanel />);

    // data_quality scores higher per-weight than it contributes: 0.8 at weight
    // 0.05 moves the total less than statistical_rarity at 0.25.
    expect(await screen.findByText(/0\.2250/)).toBeInTheDocument();
    expect(screen.getByText(/0\.0400/)).toBeInTheDocument();
  });

  it("says the total was renormalised when evidence was missing", async () => {
    respond({ engine_candidates: { count: 1, candidates: [EXPLAINED_CANDIDATE] } });
    render(<ExplainPanel />);

    expect(await screen.findByText(/weight used 0\.85/i)).toBeInTheDocument();
    expect(screen.getByText(/renormalised over the weight that was available/i))
      .toBeInTheDocument();
  });

  it("gives the reason an artifact indicator fired, not just the likelihood", async () => {
    respond({ engine_candidates: { count: 1, candidates: [EXPLAINED_CANDIDATE] } });
    render(<ExplainPanel />);

    expect(await screen.findByText(/Period matches the 0\.5 d cadence/i)).toBeInTheDocument();
    expect(screen.getByText(/Detected independently in Gaia/i)).toBeInTheDocument();
  });

  it("says no indicators fired rather than showing an empty table", async () => {
    const clean = {
      ...EXPLAINED_CANDIDATE,
      artifact: { likelihood: 0.0, verdict: "no strong artifact indicators", indicators: [], clearing_evidence: [] },
    };
    respond({ engine_candidates: { count: 1, candidates: [clean] } });
    render(<ExplainPanel />);

    expect(await screen.findByText(/No artifact indicators fired/i)).toBeInTheDocument();
  });
});

describe("SkyExplorer 3D toggle", () => {
  it("stays on the 2D map until the 3D toggle is clicked", async () => {
    respond();
    render(<SkyExplorer />);

    // Aladin's dynamic import fails gracefully in this test environment.
    expect(
      await screen.findByText(/Sky map unavailable offline|remote HiPS tiles/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/plotted · drag to orbit/i)).not.toBeInTheDocument();
  });

  it("reports how many candidates lack a reliable Gaia distance", async () => {
    respond({
      engine_candidates_spatial: {
        points: [
          { candidate_id: "c1", ra_deg: 10, dec_deg: 20, gaia_distance_pc: 200,
            gaia_abs_g_mag: 5.0, gaia_parallax_snr: 40, distance_reliable: true,
            score_total: 0.6 },
          { candidate_id: "c2", ra_deg: 11, dec_deg: 21, gaia_distance_pc: null,
            gaia_abs_g_mag: null, gaia_parallax_snr: null, distance_reliable: false,
            score_total: 0.3 },
        ],
        total: 2, reliable: 1, snr_threshold: 5.0, gaia_matched: 1, gaia_match_rate: 0.5,
      },
    });
    render(<SkyExplorer />);

    fireEvent.click(await screen.findByRole("button", { name: /^3D$/i }));

    expect(
      await screen.findByText(/1 of 2 candidates plotted/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/1 lack a reliable Gaia distance/i)).toBeInTheDocument();
  });

  it("surfaces an engine error for the spatial query without breaking the map", async () => {
    respond({ engine_candidates_spatial: new Error("engine not running") });
    render(<SkyExplorer />);

    fireEvent.click(await screen.findByRole("button", { name: /^3D$/i }));

    expect(await screen.findByText(/engine not running/i)).toBeInTheDocument();
  });

  it("shows an empty spatial result as zero of zero, not a crash", async () => {
    respond();
    render(<SkyExplorer />);

    fireEvent.click(await screen.findByRole("button", { name: /^3D$/i }));

    expect(await screen.findByText(/0 of 0 candidates plotted/i)).toBeInTheDocument();
  });
});

describe("GwCoincidence", () => {
  const baseCandidate = {
    candidate_id: "cand-1", rank: 1, object_id: "ZTF18abc", survey: "ZTF",
    release: "dr24", band: "g", path: "curves/ztf/z1.parquet",
    ra_deg: 180.122, dec_deg: 22.411,
    score: { total: 0.5 }, artifact: {}, features: {}, explanation: {},
  };

  it("says no check has run yet when there is no gw evidence", () => {
    render(<GwCoincidence candidate={baseCandidate as never} />);

    expect(
      screen.getByText(/No GW coincidence check has been run/i),
    ).toBeInTheDocument();
  });

  it("says no event overlapped the window when nothing was temporally coincident", () => {
    const candidate = {
      ...baseCandidate,
      gw: { checked_events: 5, temporally_coincident: 0, coincident: [],
           state: "no_match" as const, window_days: 30 },
    };
    render(<GwCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/5 events checked/i)).toBeInTheDocument();
    expect(screen.getByText(/No event overlapped/i)).toBeInTheDocument();
  });

  it("shows a coincident event and marks it as not part of the score", () => {
    const candidate = {
      ...baseCandidate,
      gw: {
        checked_events: 3, temporally_coincident: 1,
        coincident: [{
          event: "GW170817-v3", catalog: "GWTC-1-confident", gps_time: 1187008882.4,
          probability_density: 0.02, credible_level: 0.15, in_90pct_region: true,
          position_source: "gw_posterior" as const,
        }],
        state: "match" as const, window_days: 30,
      },
    };
    render(<GwCoincidence candidate={candidate as never} />);

    expect(screen.getByText("GW170817-v3")).toBeInTheDocument();
    expect(screen.getByText(/coincidence found/i)).toBeInTheDocument();
    expect(screen.getByText(/Not part of this candidate's score/i)).toBeInTheDocument();
  });

  it("distinguishes an EM-counterpart-fixed position from a real GW posterior", () => {
    const candidate = {
      ...baseCandidate,
      gw: {
        checked_events: 1, temporally_coincident: 1,
        coincident: [{
          event: "GW170817-v3", catalog: "GWTC-1-confident", gps_time: 1187008882.4,
          probability_density: 1.0, credible_level: 1.0, in_90pct_region: false,
          position_source: "em_counterpart_fixed" as const,
        }],
        state: "no_match" as const, window_days: 30,
      },
    };
    render(<GwCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/fixed to known counterpart/i)).toBeInTheDocument();
  });

  it("surfaces the unavailable reason rather than an empty table", () => {
    const candidate = {
      ...baseCandidate,
      gw: { checked_events: 0, coincident: [], state: "unavailable" as const,
           reason: "candidate light curve unreadable or empty" },
    };
    render(<GwCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/light curve unreadable/i)).toBeInTheDocument();
  });
});

describe("FrbCoincidence", () => {
  const baseCandidate = {
    candidate_id: "cand-1", rank: 1, object_id: "ZTF18abc", survey: "ZTF",
    release: "dr24", band: "g", path: "curves/ztf/z1.parquet",
    ra_deg: 180.122, dec_deg: 22.411,
    score: { total: 0.5 }, artifact: {}, features: {}, explanation: {},
  };

  it("says no check has run yet when there is no frb evidence", () => {
    render(<FrbCoincidence candidate={baseCandidate as never} />);

    expect(
      screen.getByText(/No FRB coincidence check has been run/i),
    ).toBeInTheDocument();
  });

  it("says no burst overlapped the window when nothing was temporally coincident", () => {
    const candidate = {
      ...baseCandidate,
      frb: { checked_bursts: 8, temporally_coincident: 0, coincident: [],
            state: "no_match" as const, window_days: 1, sigma_threshold: 3 },
    };
    render(<FrbCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/8 bursts checked/i)).toBeInTheDocument();
    expect(screen.getByText(/No burst overlapped/i)).toBeInTheDocument();
  });

  it("shows a coincident burst by its sigma offset for the common ellipse case", () => {
    const candidate = {
      ...baseCandidate,
      frb: {
        checked_bursts: 2, temporally_coincident: 1,
        coincident: [{
          burst: "FRB20200101A", repeater_name: "", mjd_400: 58800.0,
          sigma_offset: 0.8, sigma_threshold: 3.0, position_source: "ellipse" as const,
        }],
        state: "match" as const, window_days: 1, sigma_threshold: 3,
      },
    };
    render(<FrbCoincidence candidate={candidate as never} />);

    expect(screen.getByText("FRB20200101A")).toBeInTheDocument();
    expect(screen.getByText(/0\.80σ/)).toBeInTheDocument();
    expect(screen.getByText(/error ellipse/i)).toBeInTheDocument();
    expect(screen.getByText(/Not part of this candidate's score/i)).toBeInTheDocument();
  });

  it("shows a confidence level for the baseband-localized case", () => {
    const candidate = {
      ...baseCandidate,
      frb: {
        checked_bursts: 1, temporally_coincident: 1,
        coincident: [{
          burst: "FRB20200102A", repeater_name: "", mjd_400: 58801.0,
          sigma_offset: 0.1, sigma_threshold: 3.0, position_source: "healpix" as const,
          confidence_level: 0.42, in_90pct_region: true,
        }],
        state: "match" as const, window_days: 1, sigma_threshold: 3,
      },
    };
    render(<FrbCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/CL 0\.420/)).toBeInTheDocument();
    expect(screen.getByText(/baseband localization/i)).toBeInTheDocument();
  });

  it("surfaces the unavailable reason rather than an empty table", () => {
    const candidate = {
      ...baseCandidate,
      frb: { checked_bursts: 0, coincident: [], state: "unavailable" as const,
            reason: "candidate light curve unreadable or empty" },
    };
    render(<FrbCoincidence candidate={candidate as never} />);

    expect(screen.getByText(/light curve unreadable/i)).toBeInTheDocument();
  });
});

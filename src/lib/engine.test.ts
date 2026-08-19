/** Argument marshalling for the Tauri bridge.
 *
 * These wrappers are the one place where a typo is invisible. Tauri matches
 * command arguments by NAME, so `radiusArcsec` sent as `radius_arcsec` does not
 * raise — it arrives as `None` and the engine quietly uses a default. The tests
 * below pin the names and the defaults that the Rust commands expect.
 */
import { describe, expect, it } from "vitest";

import { invoke } from "@/test/setup";
import { engine } from "@/lib/engine";

function lastCall() {
  const calls = invoke.mock.calls;
  return calls[calls.length - 1] as [string, Record<string, unknown>];
}

describe("command names", () => {
  it("every wrapper targets an engine_ command", async () => {
    invoke.mockResolvedValue({});
    const calls: Array<Promise<unknown>> = [
      engine.ping(),
      engine.hardware(),
      engine.experiments(),
      engine.crossmatch(),
      engine.labelSummary(),
      engine.featureNames(),
    ];
    await Promise.all(calls);

    for (const [command] of invoke.mock.calls) {
      expect(command).toMatch(/^engine_[a-z0-9_]+$/);
    }
  });
});

describe("camelCase argument names", () => {
  it("acquire sends the names the Rust command declares", async () => {
    invoke.mockResolvedValue({});
    await engine.acquire({
      raDeg: 180.122, decDeg: 22.411, radiusArcsec: 10,
      surveys: ["ztf"], limit: 50,
    });

    const [command, args] = lastCall();
    expect(command).toBe("engine_acquire");
    expect(Object.keys(args)).toEqual(
      expect.arrayContaining(["raDeg", "decDeg", "radiusArcsec", "surveys", "limit"]),
    );
    expect(args.raDeg).toBe(180.122);
  });

  it("curveGet passes maxPoints and the canonical frame", async () => {
    invoke.mockResolvedValue({});
    await engine.curveGet("/a/b.parquet", 500, "BJD_TDB");

    const [, args] = lastCall();
    expect(args).toMatchObject({
      path: "/a/b.parquet", maxPoints: 500, frame: "BJD_TDB",
    });
  });

  it("experimentCompare sends an id list and a metric", async () => {
    invoke.mockResolvedValue({ rows: [] });
    await engine.experimentCompare(["EXP-0001", "EXP-0002"], "average_precision");

    const [command, args] = lastCall();
    expect(command).toBe("engine_experiment_compare");
    expect(args.experimentIds).toEqual(["EXP-0001", "EXP-0002"]);
    expect(args.metric).toBe("average_precision");
  });
});

describe("ablation stratification", () => {
  it("defaults to unstratified, preserving the previous behaviour", async () => {
    invoke.mockResolvedValue({});
    await engine.ablation();

    const [, args] = lastCall();
    expect(args).toMatchObject({ fraction: 0.1, seed: 42 });
    expect(args.survey).toBeUndefined();
  });

  it("forwards the survey when one is chosen", async () => {
    invoke.mockResolvedValue({});
    await engine.ablationRepeated(0.2, [1, 2], "ztf");

    const [command, args] = lastCall();
    expect(command).toBe("engine_ablation_repeated");
    expect(args).toMatchObject({ fraction: 0.2, seeds: [1, 2], survey: "ztf" });
  });
});

describe("deep models", () => {
  it("sweep defaults to three seeds so an interval can be reported", async () => {
    invoke.mockResolvedValue({});
    await engine.deepSweep();

    const [command, args] = lastCall();
    expect(command).toBe("engine_deep_sweep");
    expect(args.seeds).toEqual([17, 29, 43]);
    expect((args.seeds as number[]).length).toBeGreaterThan(1);
  });

  it("propagates the engine's explanation when torch is absent", async () => {
    invoke.mockRejectedValue(
      "PyTorch is not available in this build, so deep models cannot run.",
    );
    await expect(engine.deepTrain()).rejects.toContain("PyTorch is not available");
  });
});

describe("credentials", () => {
  it("sends the key but never asks for one back", async () => {
    invoke.mockResolvedValue({ configured: true, backend: "dpapi" });
    await engine.tnsCredentialsConfigure("secret-key", "42", "ASTRA");

    const [command, args] = lastCall();
    expect(command).toBe("engine_tns_credentials_configure");
    expect(args.apiKey).toBe("secret-key");

    invoke.mockResolvedValue({ tns_credentials: { configured: true, backend: "dpapi" } });
    const status = await engine.catalogStatus();
    expect(JSON.stringify(status)).not.toContain("secret-key");
  });
});

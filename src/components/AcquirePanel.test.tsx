/** Regression test for a real reported bug: clicking "Run acquisition" a
 *  second time with the same parameters appeared to do nothing.
 *
 * Root cause: `jobs.submit()`'s idempotency check returns
 * `{job_id, status, method, existing: true}` for a repeat of the same key --
 * no `result` field -- when an earlier job with that key isn't failed or
 * cancelled. If that earlier job already completed, the returned status is
 * already terminal, so the polling effect (which skips terminal jobs) never
 * fires and `setResult` was never called. The fix fetches the full record via
 * `jobStatus` right after submit, which does carry the result.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { invoke } from "@/test/setup";
import { AcquirePanel } from "@/components/AcquirePanel";
import type { AcquisitionResult, SurveyInfo } from "@/lib/engine";

const SURVEYS: SurveyInfo[] = [
  { name: "ZTF", release: "dr24", class: "ztf" },
  { name: "Gaia", release: "dr3", class: "gaia" },
];

const RESULT: AcquisitionResult = {
  dataset_id: "ds_1",
  project_id: null,
  query: { ra_deg: 291.3663, dec_deg: 42.7844, radius_arcsec: 10 },
  surveys: [{
    survey: "ZTF", release: "dr24", sources_found: 4, curves_stored: 4,
    points_stored: 800, mb_stored: 0.05, skipped_existing: 0,
  }],
  totals: { curves: 4, points: 800, mb: 0.05 },
  manifest_path: "/manifests/ds_1.json",
  content_hash: "abc123",
};

function respond(table: Record<string, unknown>) {
  invoke.mockImplementation((command: string) => {
    if (!(command in table)) return Promise.resolve({});
    const value = table[command];
    return value instanceof Error ? Promise.reject(value) : Promise.resolve(value);
  });
}

describe("AcquirePanel idempotent replay", () => {
  it("shows the result immediately when jobSubmit replays an already-completed job", async () => {
    respond({
      engine_job_submit: { job_id: "job_1", status: "completed", method: "acquire.cone", existing: true },
      engine_job_status: { job_id: "job_1", method: "acquire.cone", status: "completed", result: RESULT },
    });

    render(<AcquirePanel surveys={SURVEYS} />);
    fireEvent.click(await screen.findByRole("button", { name: /run acquisition/i }));

    // The result table must appear -- this is the bug: it silently didn't.
    expect(await screen.findByText(/ds_1/)).toBeInTheDocument();
    expect(screen.getByText(/Run acquisition/i)).toBeInTheDocument(); // button re-enabled, not stuck "running"
  });

  it("shows the error immediately when jobSubmit replays an already-failed-then-retried job", async () => {
    // failed/cancelled jobs are NOT replayed by submit() (a fresh job_id is
    // issued instead), but a job that fails between submit and the first
    // status fetch must still surface its error without waiting for a poll.
    respond({
      engine_job_submit: { job_id: "job_2", status: "failed", method: "acquire.cone" },
      engine_job_status: { job_id: "job_2", method: "acquire.cone", status: "failed", error: "network unreachable" },
    });

    render(<AcquirePanel surveys={SURVEYS} />);
    fireEvent.click(await screen.findByRole("button", { name: /run acquisition/i }));

    expect(await screen.findByText(/network unreachable/i)).toBeInTheDocument();
  });

  it("still polls normally for a genuinely fresh job", async () => {
    let statusCalls = 0;
    invoke.mockImplementation((command: string) => {
      if (command === "engine_job_submit") {
        return Promise.resolve({ job_id: "job_3", status: "queued", method: "acquire.cone" });
      }
      if (command === "engine_job_status") {
        statusCalls += 1;
        return Promise.resolve(
          statusCalls === 1
            ? { job_id: "job_3", method: "acquire.cone", status: "queued" }
            : { job_id: "job_3", method: "acquire.cone", status: "completed", result: RESULT },
        );
      }
      return Promise.resolve({});
    });

    render(<AcquirePanel surveys={SURVEYS} />);
    fireEvent.click(await screen.findByRole("button", { name: /run acquisition/i }));

    expect(await screen.findByText(/acquisition running/i)).toBeInTheDocument();
    expect(await screen.findByText(/ds_1/)).toBeInTheDocument();
  });
});

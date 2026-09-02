import { useState } from "react";
import { Clock3, Telescope } from "lucide-react";

import { engine, type Candidate, type FollowupRequestEntry } from "@/lib/engine";
import { Badge, Button, Field, KeyValue, Note, Table } from "@/components/ui";

export function FollowupDraft({ candidate, projectId }: { candidate: Candidate; projectId?: string }) {
  const [latitude, setLatitude] = useState("43.65");
  const [longitude, setLongitude] = useState("-79.38");
  const [duration, setDuration] = useState("12");
  const [minimumAltitude, setMinimumAltitude] = useState("30");
  const [plan, setPlan] = useState<Awaited<ReturnType<typeof engine.followupPlan>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [facilityName, setFacilityName] = useState("");
  const [requestNote, setRequestNote] = useState("");
  const [history, setHistory] = useState<FollowupRequestEntry[] | null>(null);
  const [trackingBusy, setTrackingBusy] = useState(false);
  const [trackingError, setTrackingError] = useState<string | null>(null);

  async function loadHistory() {
    setTrackingBusy(true);
    setTrackingError(null);
    try {
      setHistory(await engine.followupHistory(candidate.candidate_id, projectId));
    } catch (err) {
      setTrackingError(String(err));
    } finally {
      setTrackingBusy(false);
    }
  }

  async function requestFollowup() {
    setTrackingBusy(true);
    setTrackingError(null);
    try {
      const entry = await engine.followupRequest(candidate.candidate_id, facilityName, requestNote, projectId);
      setHistory((current) => [...(current ?? []), entry]);
      setFacilityName("");
      setRequestNote("");
    } catch (err) {
      setTrackingError(String(err));
    } finally {
      setTrackingBusy(false);
    }
  }

  async function recordResult(requestId: string, status: "observed" | "no_show" | "cancelled") {
    setTrackingBusy(true);
    setTrackingError(null);
    try {
      const updated = await engine.followupResult(requestId, status, "", projectId);
      setHistory((current) =>
        (current ?? []).map((entry) => (entry.request_id === requestId ? updated : entry)),
      );
    } catch (err) {
      setTrackingError(String(err));
    } finally {
      setTrackingBusy(false);
    }
  }

  async function makePlan() {
    setBusy(true);
    setError(null);
    try {
      setPlan(await engine.followupPlan({
        raDeg: candidate.ra_deg,
        decDeg: candidate.dec_deg,
        latitudeDeg: Number(latitude),
        longitudeDeg: Number(longitude),
        durationHours: Number(duration),
        minAltitudeDeg: Number(minimumAltitude),
        targetId: candidate.candidate_id,
      }));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Latitude" value={latitude} onChange={setLatitude} width="w-24" />
        <Field label="Longitude" value={longitude} onChange={setLongitude} width="w-24" />
        <Field label="Hours" value={duration} onChange={setDuration} width="w-16" />
        <Field label="Min altitude" value={minimumAltitude} onChange={setMinimumAltitude} width="w-20" />
        <Button icon={Clock3} tone="accent" disabled={busy} onClick={() => void makePlan()}>
          {busy ? "Planning…" : "Plan visibility"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {!plan && !error && <Note>Coordinates are pre-filled from the candidate. The result is a draft geometry check, not an observing request.</Note>}
      {plan && (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={plan.visible ? "ok" : "warn"}>{plan.visible ? "visible window found" : "not visible"}</Badge>
            <Badge tone="muted">{plan.windows.length} window{plan.windows.length === 1 ? "" : "s"}</Badge>
            {plan.best_slot && <Badge tone="accent">best airmass {plan.best_slot.airmass.toFixed(2)}</Badge>}
          </div>
          {plan.best_slot && (
            <KeyValue rows={[
              ["Best UTC", plan.best_slot.utc],
              ["Altitude", `${plan.best_slot.altitude_deg.toFixed(1)}°`],
              ["Azimuth", `${plan.best_slot.azimuth_deg.toFixed(1)}°`],
            ]} />
          )}
          {plan.windows.length > 0 && (
            <Table head={["Start UTC", "End UTC", "Slots"]}>
              {plan.windows.map((window) => (
                <tr key={`${window.start_utc}-${window.end_utc}`} className="border-b border-[var(--color-edge)]/50">
                  <td className="px-2 py-1.5 font-mono text-[11px]">{window.start_utc}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px]">{window.end_utc}</td>
                  <td className="px-2 py-1.5">{window.slots}</td>
                </tr>
              ))}
            </Table>
          )}
          <Note tone="warn">{plan.caveats[0]} No observation request was submitted.</Note>
        </>
      )}

      <div className="mt-2 border-t border-[var(--color-edge)]/50 pt-3">
        <p className="mb-1.5 text-[11px] font-medium text-[var(--color-muted)]">Follow-up tracking</p>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Facility" value={facilityName} onChange={setFacilityName} width="w-32" />
          <Field label="Note" value={requestNote} onChange={setRequestNote} width="w-40" />
          <Button icon={Telescope} tone="accent" disabled={trackingBusy} onClick={() => void requestFollowup()}>
            Request follow-up
          </Button>
          <Button disabled={trackingBusy} onClick={() => void loadHistory()}>
            {history ? "Refresh history" : "Load history"}
          </Button>
        </div>
        {trackingError && <Note tone="bad">{trackingError}</Note>}
        {history && history.length === 0 && <Note tone="muted">No follow-up requested yet.</Note>}
        {history && history.length > 0 && (
          <Table head={["Requested", "Facility", "Status", "Note", "Action"]}>
            {history.map((entry) => (
              <tr key={entry.request_id} className="border-b border-[var(--color-edge)]/50">
                <td className="px-2 py-1.5 font-mono text-[11px]">{entry.requested_utc}</td>
                <td className="px-2 py-1.5">{entry.facility_name || "—"}</td>
                <td className="px-2 py-1.5">
                  <Badge tone={entry.status === "requested" ? "warn" : entry.status === "observed" ? "ok" : "muted"}>
                    {entry.status.replace("_", " ")}
                  </Badge>
                </td>
                <td className="px-2 py-1.5 text-[var(--color-muted)]">{entry.result_note || entry.note || "—"}</td>
                <td className="px-2 py-1.5">
                  {entry.status === "requested" && (
                    <div className="flex flex-wrap gap-1">
                      <Button disabled={trackingBusy} onClick={() => void recordResult(entry.request_id, "observed")}>
                        Observed
                      </Button>
                      <Button disabled={trackingBusy} onClick={() => void recordResult(entry.request_id, "no_show")}>
                        No-show
                      </Button>
                      <Button disabled={trackingBusy} onClick={() => void recordResult(entry.request_id, "cancelled")}>
                        Cancel
                      </Button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </Table>
        )}
      </div>
    </div>
  );
}

import { Play, Radio, RefreshCw, Upload } from "lucide-react";
import { useMemo, useState } from "react";

import { engine, type AlertProviderInfo, type EventCluster, type EventProviderInfo } from "@/lib/engine";
import { Badge, Button, Empty, Note, Panel, Select, Table, useAction, useAsync } from "@/components/ui";

const EXAMPLE_PACKET = {
  event_id: "example-event",
  packet_id: "example-packet-1",
  event_time: "2026-01-01T00:00:00Z",
  localization: { ra_deg: 180.122, dec_deg: 22.411, error_radius_arcsec: 3 },
  classifications: [{ label: "unclassified", probability: null }],
};

function isCluster(value: EventCluster | unknown): value is EventCluster {
  return Boolean(value && typeof value === "object" && "event_id" in value && "packet_count" in value);
}

export function EventsView({ projectId }: { projectId?: string }) {
  const [provider, setProvider] = useState("generic");
  const [payload, setPayload] = useState(JSON.stringify(EXAMPLE_PACKET, null, 2));
  const providers = useAsync(() => engine.eventProviders(), []);
  const alertProviders = useAsync(() => engine.alertProviders(), []);
  const eventList = useAsync(
    () => engine.events({ projectId, limit: 200 }),
    [projectId],
  );
  const action = useAction();
  const clusters = useMemo(
    () => (eventList.data ?? []).filter(isCluster),
    [eventList.data],
  );

  const providerOptions = (providers.data ?? []).map((item: EventProviderInfo) => ({
    value: item.name,
    label: `${item.name} · ${item.kind}`,
  }));
  if (!providerOptions.some((item) => item.value === provider)) {
    providerOptions.unshift({ value: provider, label: provider });
  }

  async function ingest() {
    await action.run("Ingesting event packet…", async () => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        throw new Error("Payload must be valid JSON. VOEvent XML can be submitted through the RPC/API.");
      }
      const packet = await engine.eventIngest({ provider, payload: parsed, projectId });
      await eventList.reload();
      return `Stored ${packet.event_id} (${packet.packet_id}).`;
    });
  }

  function formatPayload() {
    try {
      setPayload(JSON.stringify(JSON.parse(payload), null, 2));
      action.setStatus("JSON formatted.");
    } catch {
      action.setStatus("Payload must be valid JSON before it can be formatted.");
    }
  }

  async function replay() {
    await action.run("Replaying indexed packets…", async () => {
      const packets = await engine.eventReplay({ provider, projectId, limit: 100 });
      return `Replay is read-only: ${packets.length} packet${packets.length === 1 ? "" : "s"} available.`;
    });
  }

  async function associate() {
    await action.run("Associating event localizations…", async () => {
      const result = await engine.eventAssociate({ projectId, provider });
      const count = Number(result.associations ?? 0);
      return `Checked ${String(result.events_checked ?? 0)} events; recorded ${count} conservative association${count === 1 ? "" : "s"}.`;
    });
  }

  async function pollAlerts() {
    const selected = (alertProviders.data ?? [])[0] as AlertProviderInfo | undefined;
    if (!selected) return;
    await action.run(`Polling ${selected.name}…`, async () => {
      const result = await engine.alertPoll({ provider: selected.name, projectId, offline: true });
      await eventList.reload();
      if (result.state === "offline") {
        return `${selected.name} is offline-safe: no cached poll was available.`;
      }
      const parts = [`Ingested ${result.ingested} ${selected.name} packet${result.ingested === 1 ? "" : "s"}.`];
      if (typeof result.duplicate_rate === "number") {
        parts.push(`Duplicate rate ${(result.duplicate_rate * 100).toFixed(0)}%.`);
      }
      if (result.latency_summary) {
        parts.push(`Latency median ${result.latency_summary.median.toFixed(0)}s (p95 ${result.latency_summary.p95.toFixed(0)}s, n=${result.latency_summary.n}).`);
      }
      return parts.join(" ");
    });
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel
        icon={Radio}
        title="Event inbox"
        description="Normalize alert, VOEvent, and multimessenger packets without mixing them into point-source survey rows."
        actions={<Button icon={RefreshCw} onClick={() => void eventList.reload()}>Refresh</Button>}
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-end gap-3">
            <Select label="Provider" value={provider} options={providerOptions} onChange={setProvider} />
            <Button icon={Upload} tone="accent" disabled={action.busy} onClick={() => void ingest()}>
              Ingest JSON packet
            </Button>
            <Button icon={Play} disabled={action.busy} onClick={() => void replay()}>
              Replay indexed packets
            </Button>
            <Button disabled={action.busy || clusters.length === 0} onClick={() => void associate()}>
              Associate with candidates
            </Button>
            <Button disabled={action.busy || !alertProviders.data?.length} onClick={() => void pollAlerts()}>
              Poll cached alerts
            </Button>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <label htmlFor="event-payload" className="text-sm font-medium text-[var(--color-text)]">Packet payload</label>
            <div className="flex gap-2">
              <Button onClick={formatPayload} disabled={action.busy}>Format JSON</Button>
              <Button onClick={() => setPayload(JSON.stringify(EXAMPLE_PACKET, null, 2))} disabled={action.busy}>Use example</Button>
            </div>
          </div>
          <label htmlFor="event-payload" className="sr-only">JSON event packet</label>
            <textarea
              id="event-payload"
              name="event-payload"
              value={payload}
              onChange={(event) => setPayload(event.target.value)}
              rows={8}
              spellCheck={false}
              aria-describedby="event-payload-help"
              className="min-h-48 w-full rounded border border-[var(--color-edge)] bg-[var(--color-void)] px-2 py-2 font-mono text-sm text-[var(--color-text)] outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40"
            />
          <p id="event-payload-help" className="text-xs text-[var(--color-muted)]">Provide one JSON object. Required fields include event ID, packet ID, event time, and localization when available.</p>
          {action.status && <Note tone={action.status.startsWith("Error") ? "bad" : "muted"}>{action.status}</Note>}
          <Note>Raw packets are content-addressed and retained for offline replay. A missing provider is reported as unavailable, never as a negative scientific result.</Note>
        </div>
      </Panel>

      <Panel icon={Radio} title="Event clusters" description="One event can contain multiple revised packets from the same provider.">
        {eventList.loading && <Empty>Loading event index…</Empty>}
        {eventList.error && <Note tone="bad">{eventList.error}</Note>}
        {!eventList.loading && !eventList.error && clusters.length === 0 && (
          <Empty>No event packets indexed yet. Ingest a JSON packet or replay a saved inbox.</Empty>
        )}
        {clusters.length > 0 && (
          <Table caption="Indexed astronomical event clusters" head={["Event", "Provider", "Packets", "Last seen", "Classification"]}>
            {clusters.map((item) => (
              <tr key={item.event_id} className="border-b border-[var(--color-edge)]/60">
                <td className="px-2 py-2 font-mono text-[11px]">{item.event_id}</td>
                <td className="px-2 py-2"><Badge tone="accent">{item.provider}</Badge></td>
                <td className="px-2 py-2 text-[var(--color-muted)]">{item.packet_count}</td>
                <td className="px-2 py-2 font-mono text-[11px] text-[var(--color-muted)]">{item.last_seen_utc}</td>
                <td className="px-2 py-2 text-[var(--color-muted)]">
                  {item.classifications[0]?.label ?? "unclassified"}
                </td>
              </tr>
            ))}
          </Table>
        )}
      </Panel>
    </div>
  );
}

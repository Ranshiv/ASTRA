/** Settings: credentials, storage and engine paths.
 *
 * The TNS API key is written straight to the engine, which protects it with
 * Windows DPAPI. It is never read back into the interface — `catalogStatus`
 * reports only whether a key is configured and usable, which is all the UI
 * needs to know and all it should be able to see.
 */
import { Database, HardDrive, KeyRound, Trash2 } from "lucide-react";
import { useState, type MouseEvent } from "react";

import { engine } from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Table, num, useAction, useAsync } from "@/components/ui";

export function SettingsView() {
  const catalog = useAsync(() => engine.catalogStatus());
  const paths = useAsync(() => engine.paths());
  const usage = useAsync(() => engine.storeUsage());
  const cache = useAsync(() => engine.cacheStatus());
  const readiness = useAsync(() => engine.readiness());

  const [apiKey, setApiKey] = useState("");
  const [botId, setBotId] = useState("");
  const [botName, setBotName] = useState("ASTRA");
  const credentials = useAction();
  const storage = useAction();

  const dataset = usage.data?.dataset;

  const SECTIONS = [
    { id: "settings-readiness", label: "Readiness" },
    { id: "settings-tns", label: "TNS credentials" },
    { id: "settings-storage", label: "Storage" },
    { id: "settings-paths", label: "Engine paths" },
  ];

  function jumpToSection(event: MouseEvent<HTMLAnchorElement>, id: string) {
    event.preventDefault();
    window.history.replaceState(null, "", "#/settings");
    document.getElementById(id)?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="flex flex-col gap-6">
      <nav aria-label="Settings sections" className="flex flex-wrap gap-2 text-xs">
        {SECTIONS.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            onClick={(event) => jumpToSection(event, section.id)}
            className="rounded-full border border-[var(--color-edge)] px-2.5 py-1 text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
          >
            {section.label}
          </a>
        ))}
      </nav>

      <Panel icon={KeyRound} title={<span id="settings-readiness">Capability readiness</span>} description="External gates are reported without exposing secrets or pretending unavailable resources are active.">
        {readiness.data ? <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded border border-[var(--color-edge)] p-2 text-xs"><p className="font-medium">Gaia epoch photometry</p><p className="mt-1 text-[var(--color-muted)]">{readiness.data.gaia_epoch.status} · expected {readiness.data.gaia_epoch.expected_release} · pipeline {readiness.data.gaia_epoch.code_ready ? "ready" : "not built"}</p></div>
          <div className="rounded border border-[var(--color-edge)] p-2 text-xs"><p className="font-medium">Multimodal training</p><p className="mt-1 text-[var(--color-muted)]">{readiness.data.multimodal.status} · {readiness.data.multimodal.free_vram_mb ?? "—"} MB free</p></div>
          <div className="rounded border border-[var(--color-edge)] p-2 text-xs"><p className="font-medium">Production release</p><p className="mt-1 text-[var(--color-muted)]">{readiness.data.release.status}</p></div>
        </div> : <Empty>{readiness.error ?? "Checking readiness…"}</Empty>}
      </Panel>

      <Panel
        icon={KeyRound}
        title={<span id="settings-tns">Transient Name Server credentials</span>}
        description="Optional. SIMBAD and VSX need no credentials; only TNS does."
        actions={
          <>
            <Button
              icon={KeyRound}
              tone="accent"
              disabled={credentials.busy || !apiKey.trim()}
              onClick={() =>
                void credentials.run("Storing the key with Windows DPAPI…", async () => {
                  const result = await engine.tnsCredentialsConfigure(
                    apiKey.trim(), botId.trim(), botName.trim() || "ASTRA",
                  );
                  setApiKey("");
                  await catalog.reload();
                  return `Stored via ${result.backend}. The key is never read back into this window.`;
                })
              }
            >
              Save
            </Button>
            <Button
              icon={Trash2}
              disabled={credentials.busy}
              onClick={() =>
                void (window.confirm("Clear the stored TNS credentials? You will need to enter them again for TNS queries.") && credentials.run("Clearing stored credentials…", async () => {
                  const { cleared } = await engine.tnsCredentialsClear();
                  await catalog.reload();
                  return cleared ? "Credentials cleared." : "No credentials were stored.";
                }))
              }
            >
              Clear
            </Button>
          </>
        }
      >
        <div className="flex flex-wrap items-end gap-3">
          <Field label="API key" value={apiKey} onChange={setApiKey} type="password" width="w-64" />
          <Field label="Bot ID" value={botId} onChange={setBotId} width="w-28" />
          <Field label="Bot name" value={botName} onChange={setBotName} width="w-32" />
        </div>
        {credentials.status && <Note>{credentials.status}</Note>}
        {catalog.data && (
          <div className="mt-3 flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={catalog.data.tns_credentials.configured ? "ok" : "muted"}>
                {catalog.data.tns_credentials.configured ? "configured" : "not configured"}
              </Badge>
              {catalog.data.tns_credentials.usable !== undefined && (
                <Badge tone={catalog.data.tns_credentials.usable ? "ok" : "warn"}>
                  {catalog.data.tns_credentials.usable ? "usable" : "unusable"}
                </Badge>
              )}
              <span className="text-xs text-[var(--color-muted)]" title="The secure storage backend used for this credential.">
                backend: {catalog.data.tns_credentials.backend}
              </span>
            </div>
            <h3 className="text-xs font-medium text-[var(--color-muted)]">
              Catalogue cache · {catalog.data.ttl_days}-day TTL
            </h3>
            {catalog.data.cache.entries.length === 0 ? (
              <Empty>Nothing cached yet.</Empty>
            ) : (
              <Table head={["Provider", "State", "Entries", "Earliest expiry"]}>
                {catalog.data.cache.entries.map((entry, index) => (
                  <tr key={`${entry.provider}-${entry.status}-${index}`} className="border-b border-[var(--color-edge)]/50">
                    <td className="px-2 py-1.5 font-mono">{entry.provider}</td>
                    <td className="px-2 py-1.5">{entry.status}</td>
                    <td className="px-2 py-1.5">{entry.count}</td>
                    <td className="px-2 py-1.5 text-[var(--color-muted)]">
                      {entry.earliest_expiry ?? "—"}
                    </td>
                  </tr>
                ))}
              </Table>
            )}
          </div>
        )}
      </Panel>

      <Panel
        icon={HardDrive}
        title={<span id="settings-storage">Storage</span>}
        description="Existing data is never auto-evicted; acquisition reports capacity refusals instead."
        actions={
          <>
            <Button
              disabled={storage.busy}
              onClick={() =>
                void storage.run("Enforcing the download-cache cap…", async () => {
                  const result = await engine.cacheEnforce();
                  await cache.reload();
                  return `${num(result.evicted_gb, 3)} GB across ${result.evicted_files} files evicted from the cache.`;
                })
              }
            >
              Enforce cache cap
            </Button>
            <Button
              onClick={() => {
                void usage.reload();
                void cache.reload();
              }}
            >
              Refresh
            </Button>
          </>
        }
      >
        {storage.status && <Note>{storage.status}</Note>}
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">Datasets</h3>
            {dataset ? (
              <KeyValue
                rows={[
                  ["Used", `${num(dataset.used_gb, 3)} GB`],
                  ["Cap", `${num(dataset.cap_gb, 1)} GB`],
                  ["Available", `${num(dataset.available_gb, 3)} GB`],
                  ["Usage", `${num(dataset.usage_fraction * 100, 1)}%`],
                ]}
              />
            ) : (
              <Empty>{usage.error ?? "Reading dataset usage…"}</Empty>
            )}
            {usage.data && Object.keys(usage.data.surveys).length > 0 && (
              <Table head={["Survey", "Curves", "Size"]}>
                {Object.entries(usage.data.surveys).map(([survey, stats]) => (
                  <tr key={survey} className="border-b border-[var(--color-edge)]/50">
                    <td className="px-2 py-1.5 font-mono">{survey}</td>
                    <td className="px-2 py-1.5">{stats.curves.toLocaleString()}</td>
                    <td className="px-2 py-1.5">{num(stats.gb, 4)} GB</td>
                  </tr>
                ))}
              </Table>
            )}
          </div>
          <div>
            <h3 className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">Download cache</h3>
            {cache.data ? (
              <KeyValue
                rows={[
                  ["Total", `${num(cache.data.total_gb, 3)} GB`],
                  ["Cap", `${num(cache.data.cap_gb, 1)} GB`],
                  ["Files", cache.data.file_count.toLocaleString()],
                  ["Evicted", `${num(cache.data.evicted_gb, 3)} GB / ${cache.data.evicted_files} files`],
                ]}
              />
            ) : (
              <Empty>{cache.error ?? "Reading cache status…"}</Empty>
            )}
          </div>
        </div>
      </Panel>

      <Panel icon={Database} title={<span id="settings-paths">Engine paths</span>} description="Where ASTRA keeps everything on this machine.">
        {paths.data ? (
          <KeyValue rows={Object.entries(paths.data).map(([key, value]) => [key, value])} />
        ) : (
          <Empty>{paths.error ?? "Reading paths…"}</Empty>
        )}
      </Panel>
    </div>
  );
}

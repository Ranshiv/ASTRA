import { useState } from "react";
import { BookOpen } from "lucide-react";

import { engine, type Candidate } from "@/lib/engine";
import { Badge, Button, Empty, Note, Table } from "@/components/ui";

export function LiteratureContext({ candidate, projectId }: { candidate: Candidate; projectId?: string }) {
  const [result, setResult] = useState(candidate.literature ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function search() {
    setBusy(true);
    setError(null);
    try {
      const response = await engine.literatureSearch({
        objectId: candidate.object_id,
        terms: candidate.explanation.resembles ?? [],
        eventIds: candidate.event_ids ?? [],
        projectId,
      });
      setResult(response);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const records = result?.records ?? [];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result?.complete ? "ok" : "muted"}>
          {result ? `${records.length} record${records.length === 1 ? "" : "s"}` : "not queried"}
        </Badge>
        {result && <Badge tone="muted">{result.complete ? "providers complete" : "partial / unavailable"}</Badge>}
        <Button icon={BookOpen} disabled={busy} onClick={() => void search()}>
          {busy ? "Searching…" : "Search literature"}
        </Button>
      </div>
      {error && <Note tone="bad">{error}</Note>}
      {records.length === 0 && !error && (
        <Empty>{result ? "No matching records were returned; this is not a novelty claim." : "No literature search has been run for this candidate."}</Empty>
      )}
      {records.length > 0 && (
        <Table head={["Year", "Title", "Provider"]}>
          {records.slice(0, 8).map((record, index) => (
            <tr key={`${record.url ?? record.title ?? "record"}-${index}`} className="border-b border-[var(--color-edge)]/50">
              <td className="px-2 py-1.5">{record.year ?? "—"}</td>
              <td className="px-2 py-1.5">
                {record.url ? <a href={record.url} target="_blank" rel="noreferrer" className="text-[var(--color-accent)]">{record.title ?? record.url}</a> : record.title ?? "—"}
              </td>
              <td className="px-2 py-1.5 text-[var(--color-muted)]">{record.provider}</td>
            </tr>
          ))}
        </Table>
      )}
      <Note tone="warn">Literature provides context and provenance only; it never changes the ASTRA score or treats an unavailable provider as a no-match.</Note>
    </div>
  );
}

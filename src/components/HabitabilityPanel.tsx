/** Habitable-zone position and Earth Similarity Index for a confirmed
 *  planet (roadmap: astrophysics & extraterrestrial-study feature pass).
 *
 *  Both quantities come back from `habitability.score`/`habitability.rank`,
 *  read-only diagnostic RPC methods that never write into a candidate's
 *  score -- the same "diagnostic evidence only" category
 *  `physical.characterize`/`digital_twin.*` already occupy.
 *
 *  Two caveats are surfaced here as persistent warnings, not buried in a
 *  docstring: ESI's surface term is built from equilibrium temperature
 *  (no greenhouse model exists in this engine), and ESI is a geometric
 *  similarity metric to Earth, not a probability of habitability or life.
 */
import { useState } from "react";
import { Globe, ListOrdered } from "lucide-react";

import {
  engine,
  type HabitabilityRanking,
  type HabitabilityScore,
} from "@/lib/engine";
import { Badge, Button, Empty, Field, KeyValue, Note, Panel, Table, num } from "@/components/ui";

function esiTone(value: number | null): "ok" | "warn" | "bad" | "muted" {
  if (value === null) return "muted";
  if (value >= 0.8) return "ok";
  if (value >= 0.5) return "warn";
  return "bad";
}

function ScoreDetail({ result }: { result: HabitabilityScore }) {
  const zone = result.habitable_zone;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={result.in_conservative_hz ? "ok" : "muted"}>
          {result.in_conservative_hz ? "in conservative HZ" : "outside conservative HZ"}
        </Badge>
        <Badge tone={result.in_optimistic_hz ? "ok" : "muted"}>
          {result.in_optimistic_hz ? "in optimistic HZ" : "outside optimistic HZ"}
        </Badge>
        <Badge tone={esiTone(result.esi_global)}>ESI {num(result.esi_global, 3)}</Badge>
        {zone.extrapolated && <Badge tone="warn">Teff outside 2600-7200 K</Badge>}
      </div>
      <KeyValue
        rows={[
          ["HZ position (0=inner, 1=outer)", num(result.hz_position, 3)],
          ["Conservative HZ (AU)", `${num(zone.conservative_inner_au, 3)} - ${num(zone.conservative_outer_au, 3)}`],
          ["Optimistic HZ (AU)", `${num(zone.optimistic_inner_au, 3)} - ${num(zone.optimistic_outer_au, 3)}`],
          ["Equilibrium temperature (K)", num(result.equilibrium_temp_k, 1)],
          ["ESI, interior", num(result.esi_interior, 3)],
          ["ESI, surface (from T_eq)", num(result.esi_surface_from_teq, 3)],
        ]}
      />
      <Note tone="warn">
        ESI's surface term is built from equilibrium temperature, not a measured surface
        temperature -- this engine has no greenhouse model. ESI is a geometric similarity
        score to Earth, not a probability of habitability or life, and neither value here
        is folded into candidate ranking.
      </Note>
      {result.warnings.length > 0 && (
        <div className="flex flex-col gap-1">
          {result.warnings.map((w) => (
            <Note key={w} tone="muted">{w}</Note>
          ))}
        </div>
      )}
    </div>
  );
}

function RankingTable({ result }: { result: HabitabilityRanking }) {
  if (result.planets.length === 0) {
    return <Empty>No planets in the archive matched these bounds.</Empty>;
  }
  return (
    <Table head={["Planet", "Host", "ESI", "Conservative HZ", "T_eq (K)"]}>
      {result.planets.map((planet) => (
        <tr key={`${planet.host_name}-${planet.planet_name}`} className="border-b border-[var(--color-edge)]/50">
          <td className="px-2 py-1.5">{planet.planet_name}</td>
          <td className="px-2 py-1.5">{planet.host_name}</td>
          <td className="px-2 py-1.5">
            <Badge tone={esiTone(planet.esi_global)}>{num(planet.esi_global, 3)}</Badge>
          </td>
          <td className="px-2 py-1.5">{planet.in_conservative_hz ? "yes" : "no"}</td>
          <td className="px-2 py-1.5">{num(planet.equilibrium_temp_k, 1)}</td>
        </tr>
      ))}
    </Table>
  );
}

export function HabitabilityPanel({ projectId: _projectId }: { projectId?: string }) {
  const [planetName, setPlanetName] = useState("Kepler-10 b");
  const [score, setScore] = useState<HabitabilityScore | null>(null);
  const [scoreBusy, setScoreBusy] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  const [teffMin, setTeffMin] = useState("2600");
  const [teffMax, setTeffMax] = useState("7200");
  const [ranking, setRanking] = useState<HabitabilityRanking | null>(null);
  const [rankingBusy, setRankingBusy] = useState(false);
  const [rankingError, setRankingError] = useState<string | null>(null);

  async function scorePlanet() {
    setScoreBusy(true);
    setScoreError(null);
    try {
      setScore(await engine.habitabilityScore(planetName));
    } catch (err) {
      setScoreError(String(err));
    } finally {
      setScoreBusy(false);
    }
  }

  async function rankPlanets() {
    setRankingBusy(true);
    setRankingError(null);
    try {
      setRanking(await engine.habitabilityRank({
        teffMin: Number(teffMin) || undefined,
        teffMax: Number(teffMax) || undefined,
        limit: 25,
      }));
    } catch (err) {
      setRankingError(String(err));
    } finally {
      setRankingBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel
        icon={Globe}
        title="Habitability score"
        description="Kopparapu et al. (2013) habitable-zone position and Earth Similarity Index for a confirmed planet."
        actions={
          <div className="flex items-end gap-2">
            <Field label="Planet name" value={planetName} onChange={setPlanetName} width="w-40" />
            <Button icon={Globe} disabled={scoreBusy} onClick={() => void scorePlanet()}>
              {scoreBusy ? "Scoring…" : "Score"}
            </Button>
          </div>
        }
      >
        {scoreError && <Note tone="bad">{scoreError}</Note>}
        {!score && !scoreError && (
          <Empty>No planet scored yet. Enter a confirmed planet name and score it.</Empty>
        )}
        {score && <ScoreDetail result={score} />}
      </Panel>

      <Panel
        icon={ListOrdered}
        title="Rank by ESI"
        description="Catalog-scale ranking of confirmed planets within a stellar effective-temperature band, bounded by the Kopparapu et al. (2013) validity range."
        actions={
          <div className="flex items-end gap-2">
            <Field label="Teff min (K)" value={teffMin} onChange={setTeffMin} width="w-24" />
            <Field label="Teff max (K)" value={teffMax} onChange={setTeffMax} width="w-24" />
            <Button icon={ListOrdered} disabled={rankingBusy} onClick={() => void rankPlanets()}>
              {rankingBusy ? "Ranking…" : "Rank"}
            </Button>
          </div>
        }
      >
        {rankingError && <Note tone="bad">{rankingError}</Note>}
        {!ranking && !rankingError && (
          <Empty>No ranking run yet. Set a Teff band and rank the archive.</Empty>
        )}
        {ranking && <RankingTable result={ranking} />}
      </Panel>
    </div>
  );
}

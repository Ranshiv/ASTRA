"""Research-friendly candidate exports generated inside ASTRA's data root."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from . import candidates as candidates_mod, config


def export_candidates(name: str, format: str, root: Path | None = None) -> dict:
    root = root or config.PATHS.projects
    candidates = candidates_mod.load(name, root)
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = format.lower()
    path = output / f"{name}_candidates_{stamp}.{kind}"
    if kind == "csv":
        _csv(candidates, path)
    elif kind == "fits":
        _fits(candidates, path)
    elif kind == "pdf":
        _pdf(candidates, path, name)
    else:
        raise ValueError("format must be csv, fits, or pdf")
    return {"path": str(path), "format": kind, "count": len(candidates),
            "bytes": path.stat().st_size}


def _csv(candidates, path: Path) -> None:
    fields = ("candidate_id", "rank", "survey", "release", "object_id",
              "band", "ra_deg", "dec_deg", "score", "artifact_likelihood")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({
                "candidate_id": item.candidate_id, "rank": item.rank,
                "survey": item.survey, "release": item.release,
                "object_id": item.object_id, "band": item.band,
                "ra_deg": item.ra_deg, "dec_deg": item.dec_deg,
                "score": item.score.get("total"),
                "artifact_likelihood": item.artifact.get("likelihood"),
            })


def _fits(candidates, path: Path) -> None:
    from astropy.table import Table
    rows = [{"candidate_id": c.candidate_id, "rank": c.rank,
             "survey": c.survey, "release": c.release,
             "object_id": c.object_id, "band": c.band,
             "ra_deg": c.ra_deg, "dec_deg": c.dec_deg,
             "score": c.score.get("total", 0.0)} for c in candidates]
    Table(rows=rows).write(path, format="fits", overwrite=False)


def _pdf(candidates, path: Path, name: str) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    with PdfPages(path) as pdf:
        for item in candidates:
            figure = Figure(figsize=(8.27, 11.69))
            text = [f"ASTRA candidate {item.candidate_id}",
                    f"Run: {name}    Rank: {item.rank}",
                    f"{item.survey} {item.release} / {item.object_id} / {item.band}",
                    f"Position: {item.ra_deg:.7f}, {item.dec_deg:.7f}",
                    f"Composite score: {item.score.get('total', 0):.4f}", "",
                    item.explanation.get("what_happened", ""), "",
                    "Recommended actions:",
                    *[f"• {x}" for x in item.explanation.get("recommended_actions", [])],
                    "", "Machine-readable evidence:",
                    json.dumps(item.score, indent=2)]
            figure.text(0.08, 0.94, "\n".join(text), va="top", wrap=True,
                        family="sans-serif", fontsize=10)
            pdf.savefig(figure)

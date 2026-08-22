"""Shared request/response handling for MAST CAOM position queries.

Hubble and JWST are served by the same archive and the same CAOM service; they
differ only in the `obs_collection` they filter on.  Factoring the request
shape here keeps the two connectors thin and keeps one place to fix if the
service contract moves.
"""

from __future__ import annotations

import json

from .. import netclient

INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
SERVICE = "Mast.Caom.Filtered.Position"
MAX_PAGES = 20


def caom_params(ra_deg: float, dec_deg: float, radius_deg: float,
                collection: str, limit: int, page: int = 1) -> dict[str, str]:
    """Build the query parameters for one collection-filtered cone search.

    The collection filter is applied server-side so the row cap is spent on
    the requested mission rather than on whatever else overlaps the cone.
    """
    request = {
        "service": SERVICE,
        "format": "json",
        "params": {
            "columns": "*",
            "filters": [
                {"paramName": "obs_collection", "values": [collection]},
            ],
            "position": f"{ra_deg}, {dec_deg}, {radius_deg:.8f}",
        },
        "pagesize": limit,
        "page": page,
    }
    return {"request": json.dumps(request)}


def fetch_all_pages(ra_deg: float, dec_deg: float, radius_deg: float,
                    collection: str, limit: int, provider: str = "mast",
                    timeout: int = 60, max_pages: int = MAX_PAGES) -> list[dict]:
    """Walk CAOM's native page/pagesize pagination until the cone is exhausted.

    Stops on an empty/short page, once `limit` rows are collected, or after
    `max_pages` as a safety cap. Every page still goes through
    `netclient.get`, so the provider's throttle/retry applies per page.
    """
    remaining = max(1, int(limit))
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        page_size = max(1, min(remaining, 200))
        response = netclient.get(
            INVOKE_URL,
            caom_params(ra_deg, dec_deg, radius_deg, collection, page_size, page=page),
            timeout=timeout, provider=provider,
        )
        try:
            page_rows = parse_rows(response.json(), page_size)
        except ValueError:
            page_rows = []
        if not page_rows:
            break
        rows.extend(page_rows)
        remaining -= len(page_rows)
        if len(page_rows) < page_size or remaining <= 0:
            break
    return rows


def parse_rows(payload: object, limit: int = 100) -> list[dict]:
    """Pull the row list out of a CAOM response, defensively.

    A service-level error arrives as a dict without a usable `data` list, which
    must read as "no rows" rather than as an exception.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    return [row for row in rows[:limit] if isinstance(row, dict)]

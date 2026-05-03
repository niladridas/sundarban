"""Fetch real Global Drifter Program tracks from NOAA AOML's ERDDAP.

These are physical surface drifters drogued at 15m depth, GPS-tracked, and
quality-controlled by AOML. Overlaying one of these on a simulation seeded at
the same release point/time is the most credible single validation image you
can produce.

ERDDAP (Environmental Research Division Data Access Program) returns CSV via
a simple HTTP GET — no auth required. Constraints (`time>=`, `latitude>=`,
etc.) are URL-encoded as `%3E=` / `%3C=`.

CLI:
    python -m driftscope.drifters
    python -m driftscope.drifters --region northern_bay_of_bengal --days 30
"""
from __future__ import annotations
import argparse
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

from .config import (
    DEFAULT_REGION,
    DRIFTER_ERDDAP_URL,
    DRIFTERS_DIR,
    PRESETS,
    Region,
)

console = Console()

# ERDDAP returns these exact column names for drifter_hourly_qc; if a different
# dataset is used, the renames in fetch_drifter_tracks() may need updating.
DRIFTER_FIELDS = "ID,time,latitude,longitude"


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    safe = region_name.lower().replace(" ", "_")
    return DRIFTERS_DIR / f"{safe}_{start}_{end}_drifters.csv"


def _build_url(
    bbox: tuple[float, float, float, float], start_date: str, end_date: str
) -> str:
    lon_min, lat_min, lon_max, lat_max = bbox
    GE, LE = "%3E=", "%3C="  # >= and <=, URL-encoded
    constraints = (
        f"&time{GE}{start_date}T00:00:00Z"
        f"&time{LE}{end_date}T23:59:59Z"
        f"&latitude{GE}{lat_min}"
        f"&latitude{LE}{lat_max}"
        f"&longitude{GE}{lon_min}"
        f"&longitude{LE}{lon_max}"
    )
    return f"{DRIFTER_ERDDAP_URL}?{DRIFTER_FIELDS}{constraints}"


def fetch_drifter_tracks(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Download GDP drifter positions in bbox/time, save as CSV.

    The output CSV has columns: ID, time, lat, lon.

    Empty results are normal (drifters are sparse; many bbox/time combos have
    none). An empty CSV is written so the cache check still works.
    """
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    url = _build_url(bbox, start_date, end_date)
    lon_min, lat_min, lon_max, lat_max = bbox
    console.print(
        f"[cyan]↓[/cyan] AOML GDP ERDDAP\n"
        f"   bbox: {lon_min:.2f},{lat_min:.2f} → {lon_max:.2f},{lat_max:.2f}\n"
        f"   time: {start_date} → {end_date}"
    )

    try:
        # Row 1 is units (ERDDAP convention); skip it.
        df = pd.read_csv(url, skiprows=[1])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # ERDDAP returns 404 when zero rows match the constraints.
            console.print(
                "[yellow]![/yellow] No drifters in this bbox/time. "
                "Try widening the area or shifting the dates."
            )
            empty = pd.DataFrame(columns=["ID", "time", "lat", "lon"])
            empty.to_csv(out, index=False)
            return out
        raise

    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["ID", "time"]).reset_index(drop=True)
    df.to_csv(out, index=False)

    n_drifters = df["ID"].nunique()
    console.print(
        f"[green]✓[/green] {len(df):,} positions across {n_drifters} drifters → {out.name}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Global Drifter Program tracks")
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(PRESETS.keys()))
    p.add_argument("--bbox", type=_parse_bbox, default=None,
                   help="override region: 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--days", type=int, default=30,
                   help="trailing N days (ignored if --start given)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.bbox is not None:
        bbox, name = args.bbox, "custom"
    else:
        r: Region = PRESETS[args.region]
        bbox, name = r.bbox, r.name

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        today = datetime.now(timezone.utc).date()
        end_dt = today - timedelta(days=2)
        start_dt = end_dt - timedelta(days=args.days)
        start, end = start_dt.isoformat(), end_dt.isoformat()

    fetch_drifter_tracks(bbox, start, end, region_name=name, force=args.force)


if __name__ == "__main__":
    main()

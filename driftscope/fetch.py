"""Download surface currents from Copernicus Marine for any bbox/time window.

Uses the official `copernicusmarine` Python client. Requires free CMEMS
credentials in `.env` (see .env.example).

CLI:
    python -m driftscope.fetch
    python -m driftscope.fetch --region bay_of_bengal --days 30
    python -m driftscope.fetch --bbox 88,21,90.5,22.8 --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .config import (
    CMEMS_DATASET_ID,
    CMEMS_DEPTH,
    CMEMS_VARIABLES,
    CURRENTS_DIR,
    PRESETS,
    DEFAULT_REGION,
    Region,
)

load_dotenv()
console = Console()


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    """'lon_min,lat_min,lon_max,lat_max' → tuple."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    """Deterministic filename so re-runs don't redownload."""
    safe = region_name.lower().replace(" ", "_")
    return CURRENTS_DIR / f"{safe}_{start}_{end}.nc"


def fetch_currents(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Download CMEMS surface currents for a bbox and time range.

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max)
    start_date, end_date : 'YYYY-MM-DD'
    region_name : used in the filename only
    force : redownload even if file exists

    Returns
    -------
    Path to the downloaded NetCDF.
    """
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    # Lazy import — copernicusmarine has heavy startup
    import copernicusmarine

    lon_min, lat_min, lon_max, lat_max = bbox
    console.print(
        f"[cyan]↓[/cyan] CMEMS {CMEMS_DATASET_ID}\n"
        f"   bbox: {lon_min:.2f},{lat_min:.2f} → {lon_max:.2f},{lat_max:.2f}\n"
        f"   time: {start_date} → {end_date}\n"
        f"   out:  {out}"
    )

    copernicusmarine.subset(
        dataset_id=CMEMS_DATASET_ID,
        variables=CMEMS_VARIABLES,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T00:00:00",
        minimum_depth=CMEMS_DEPTH,
        maximum_depth=CMEMS_DEPTH,
        output_filename=out.name,
        output_directory=str(out.parent),
        overwrite=True,
    )
    console.print(f"[green]✓[/green] saved {out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch CMEMS surface currents")
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(PRESETS.keys()))
    p.add_argument("--bbox", type=_parse_bbox, default=None,
                   help="override region: 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--days", type=int, default=14,
                   help="trailing N days (ignored if --start given)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.bbox is not None:
        bbox = args.bbox
        name = "custom"
    else:
        r: Region = PRESETS[args.region]
        bbox = r.bbox
        name = r.name

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        today = datetime.now(timezone.utc).date()
        end_dt = today - timedelta(days=2)  # CMEMS analysis lags ~1d
        start_dt = end_dt - timedelta(days=args.days)
        start, end = start_dt.isoformat(), end_dt.isoformat()

    if not (os.getenv("COPERNICUSMARINE_USERNAME") and
            os.getenv("COPERNICUSMARINE_PASSWORD")):
        console.print(
            "[yellow]![/yellow] CMEMS credentials not set. "
            "Copy .env.example → .env and fill in (free signup at "
            "https://marine.copernicus.eu)."
        )

    fetch_currents(bbox, start, end, region_name=name, force=args.force)


if __name__ == "__main__":
    main()

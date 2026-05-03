"""Download CMEMS Ocean Color L4 chlorophyll-a for the bbox/time window.

CMEMS Ocean Color CCI = ESA's multi-sensor (MODIS / VIIRS / OLCI) L4 daily
gap-filled chlorophyll-a, ~4 km, 1997–present. Same `copernicusmarine` auth
as currents — no extra setup.

Why chlorophyll in DriftScope?
  - For deltas/coasts, the river-plume signal in chlorophyll is **dramatic**.
    The Ganges-Hooghly outflow paints a clear "tongue" extending into the
    Bay of Bengal — directly comparable to where modeled river-tracer
    particles end up.
  - Chlorophyll fronts often co-locate with SST fronts but tell a different
    story: temperature reflects mixing/heating, chlorophyll reflects
    biology + sediment + dissolved organic matter. Two complementary
    observation channels for the same particles.
  - Chlorophyll values span 3+ orders of magnitude, so they're best viewed
    on a log colormap — this module displays log₁₀(CHL).

CLI:
    python -m driftscope.chlorophyll
    python -m driftscope.chlorophyll --region sundarbans --days 14
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .config import (
    CHL_DIR,
    CMEMS_CHL_DATASET_ID,
    CMEMS_CHL_VARIABLES,
    DEFAULT_REGION,
    PRESETS,
    Region,
)

load_dotenv()
console = Console()


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    safe = region_name.lower().replace(" ", "_")
    return CHL_DIR / f"{safe}_{start}_{end}_chl.nc"


def fetch_chlorophyll(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Download Ocean Color L4 chlorophyll for a bbox/time window. Returns NetCDF path."""
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    import copernicusmarine

    lon_min, lat_min, lon_max, lat_max = bbox
    console.print(
        f"[cyan]↓[/cyan] CMEMS {CMEMS_CHL_DATASET_ID}\n"
        f"   bbox: {lon_min:.2f},{lat_min:.2f} → {lon_max:.2f},{lat_max:.2f}\n"
        f"   time: {start_date} → {end_date}\n"
        f"   out:  {out}"
    )

    copernicusmarine.subset(
        dataset_id=CMEMS_CHL_DATASET_ID,
        variables=CMEMS_CHL_VARIABLES,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T00:00:00",
        output_filename=out.name,
        output_directory=str(out.parent),
        overwrite=True,
    )
    console.print(f"[green]✓[/green] saved {out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch CMEMS Ocean Color L4 chlorophyll")
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(PRESETS.keys()))
    p.add_argument("--bbox", type=_parse_bbox, default=None,
                   help="override region: 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--days", type=int, default=14)
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
        # Ocean Color MY product has ~6-month lag — pull 7 months back.
        today = datetime.now(timezone.utc).date()
        end_dt = today - timedelta(days=210)
        start_dt = end_dt - timedelta(days=args.days)
        start, end = start_dt.isoformat(), end_dt.isoformat()

    fetch_chlorophyll(bbox, start, end, region_name=name, force=args.force)


if __name__ == "__main__":
    main()

"""Download ERA5 10m winds for windage forcing.

Windage is the direct wind push on the air-exposed fraction of a floating
object. The standard parameterization is:

    U_object = α · U_10m

where α is a small (1–4%) empirical coefficient that depends on the object's
windage area / drag profile:

    icebergs        ~ 1%
    persons in water  ~ 2%
    plastic debris  ~ 2–4%
    oil slicks      ~ 3–3.5%

CMEMS already represents wind-driven Ekman currents in the model's surface
flow; this module adds the *additional* direct drag on the emergent fraction
of an object, on top of CMEMS + Stokes + tides.

Same `cdsapi` auth as Stokes — no extra credentials needed.

CLI:
    python -m driftscope.winds
    python -m driftscope.winds --region sundarbans --days 14
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv
from rich.console import Console

from .config import (
    DEFAULT_REGION,
    ERA5_DATASET,
    ERA5_TIMES,
    ERA5_WIND_VARIABLES,
    PRESETS,
    Region,
    WINDS_DIR,
)
from .stokes import _date_lists  # reuse — both modules query the same ERA5 catalog

load_dotenv()
console = Console()


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    safe = region_name.lower().replace(" ", "_")
    return WINDS_DIR / f"{safe}_{start}_{end}_winds.nc"


def fetch_winds(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Download ERA5 10m winds for a bbox/time window.

    Output NetCDF has variables `u10` and `v10` (m/s) on the ERA5 grid (~0.25°,
    coarser than CMEMS — same situation as waves; combine step regrids to the
    currents grid).
    """
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    import cdsapi

    lon_min, lat_min, lon_max, lat_max = bbox
    raw = out.with_suffix(".raw.nc")

    console.print(
        f"[cyan]↓[/cyan] ERA5 {ERA5_DATASET} (winds)\n"
        f"   vars: u10, v10\n"
        f"   bbox: {lon_min:.2f},{lat_min:.2f} → {lon_max:.2f},{lat_max:.2f}\n"
        f"   time: {start_date} → {end_date} ({len(ERA5_TIMES)}/day)\n"
        f"   out:  {out}"
    )

    cds = cdsapi.Client()
    cds.retrieve(
        ERA5_DATASET,
        {
            "product_type": "reanalysis",
            "variable": ERA5_WIND_VARIABLES,
            **_date_lists(start_date, end_date),
            "time": ERA5_TIMES,
            "area": [lat_max, lon_min, lat_min, lon_max],
            "data_format": "netcdf",
        },
        str(raw),
    )

    ds = xr.open_dataset(raw)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    ds.to_netcdf(out)
    raw.unlink()

    console.print(f"[green]✓[/green] saved {out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch ERA5 10m winds for windage")
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
        today = datetime.now(timezone.utc).date()
        end_dt = today - timedelta(days=6)
        start_dt = end_dt - timedelta(days=args.days)
        start, end = start_dt.isoformat(), end_dt.isoformat()

    fetch_winds(bbox, start, end, region_name=name, force=args.force)


if __name__ == "__main__":
    main()

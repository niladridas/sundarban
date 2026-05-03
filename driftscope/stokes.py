"""Download ERA5 wave parameters and derive surface Stokes drift.

ERA5 is ECMWF's hourly global reanalysis. We pull three wave variables
(significant height, mean period, mean direction) and convert them to
zonal/meridional Stokes drift components on the wave grid:

    U_s = (2π)³ · H_s² / (8 · g · T_m³)        [m/s, deep-water surface]
    direction = along the mean wave propagation (mwd is "from"; we flip)

Requires a free CDS API key — see .env.example.

CLI:
    python -m driftscope.stokes
    python -m driftscope.stokes --region northern_bay_of_bengal --days 14
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from rich.console import Console

from .config import (
    DEFAULT_REGION,
    ERA5_DATASET,
    ERA5_TIMES,
    ERA5_WAVE_VARIABLES,
    PRESETS,
    Region,
    WAVES_DIR,
)

load_dotenv()
console = Console()

G = 9.81  # m/s²


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    safe = region_name.lower().replace(" ", "_")
    return WAVES_DIR / f"{safe}_{start}_{end}_waves.nc"


def stokes_from_waves(
    swh: xr.DataArray, mwp: xr.DataArray, mwd: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute (ust, vst) from significant height, mean period, mean direction.

    `mwd` is the meteorological "from" direction (deg clockwise from N).
    We convert to a math angle for the "to" direction so cos/sin give the
    eastward/northward components: math_angle = 270° − mwd.
    """
    magnitude = (2 * np.pi) ** 3 * swh ** 2 / (8 * G * mwp ** 3)
    theta = np.deg2rad(270.0 - mwd)
    ust = magnitude * np.cos(theta)
    vst = magnitude * np.sin(theta)
    ust.attrs = {"long_name": "Surface Stokes drift, eastward", "units": "m s-1"}
    vst.attrs = {"long_name": "Surface Stokes drift, northward", "units": "m s-1"}
    return ust, vst


def _date_lists(start: str, end: str) -> dict[str, list[str]]:
    """CDS expects year/month/day as separate string lists."""
    s = datetime.fromisoformat(start).date()
    e = datetime.fromisoformat(end).date()
    days, d = [], s
    while d <= e:
        days.append(d)
        d += timedelta(days=1)
    return {
        "year": sorted({f"{x.year}" for x in days}),
        "month": sorted({f"{x.month:02d}" for x in days}),
        "day": sorted({f"{x.day:02d}" for x in days}),
    }


def fetch_stokes(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Download ERA5 waves for a bbox/time window and write Stokes drift to NetCDF.

    Returned file contains the original wave fields (swh, mwp, mwd) plus the
    derived `ust`/`vst` components on the ERA5 grid.
    """
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    import cdsapi

    lon_min, lat_min, lon_max, lat_max = bbox
    raw = out.with_suffix(".raw.nc")

    console.print(
        f"[cyan]↓[/cyan] ERA5 {ERA5_DATASET}\n"
        f"   vars: {', '.join(v.split('_')[-1] for v in ERA5_WAVE_VARIABLES)}\n"
        f"   bbox: {lon_min:.2f},{lat_min:.2f} → {lon_max:.2f},{lat_max:.2f}\n"
        f"   time: {start_date} → {end_date} ({len(ERA5_TIMES)}/day)\n"
        f"   out:  {out}"
    )

    cds = cdsapi.Client()
    cds.retrieve(
        ERA5_DATASET,
        {
            "product_type": "reanalysis",
            "variable": ERA5_WAVE_VARIABLES,
            **_date_lists(start_date, end_date),
            "time": ERA5_TIMES,
            # CDS area order: [North, West, South, East]
            "area": [lat_max, lon_min, lat_min, lon_max],
            "data_format": "netcdf",
        },
        str(raw),
    )

    ds = xr.open_dataset(raw)
    # Newer cdsapi delivers `valid_time`; older uses `time`.
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    ust, vst = stokes_from_waves(ds["swh"], ds["mwp"], ds["mwd"])
    ds["ust"], ds["vst"] = ust, vst
    ds.to_netcdf(out)
    raw.unlink()

    console.print(f"[green]✓[/green] saved {out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch ERA5 waves & compute Stokes drift")
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
        # ERA5 reanalysis lags real-time by ~5 days
        today = datetime.now(timezone.utc).date()
        end_dt = today - timedelta(days=6)
        start_dt = end_dt - timedelta(days=args.days)
        start, end = start_dt.isoformat(), end_dt.isoformat()

    if not (os.getenv("CDSAPI_URL") and os.getenv("CDSAPI_KEY")) and not Path.home().joinpath(".cdsapirc").exists():
        console.print(
            "[yellow]![/yellow] CDS API credentials not found. Either:\n"
            "  • set CDSAPI_URL + CDSAPI_KEY in .env, OR\n"
            "  • create ~/.cdsapirc (see .env.example).\n"
            "  Get a free key at https://cds.climate.copernicus.eu/api-how-to"
        )

    fetch_stokes(bbox, start, end, region_name=name, force=args.force)


if __name__ == "__main__":
    main()

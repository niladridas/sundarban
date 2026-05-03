"""Predict barotropic tidal currents from a global tide model via pyTMD.

The CMEMS global forecast filters tides out, so for any shelf/delta region
(Sundarbans, Hooghly Mouth, etc.) tidal currents are *the* dominant signal
and must be added explicitly. Tides are deterministic — given lat/lon/time
and a tide model, the current is exactly predictable.

We use **TPXO9-atlas-v5** by default (pyTMD also supports FES2014 etc).

Setup (one-time, ~1 GB):
    1. Register at https://www.tpxo.net/tpxo-products-and-registration
    2. Download TPXO9-atlas-v5 (NetCDF format)
    3. Unpack and set TIDE_MODEL_DIR in .env to that directory

CLI:
    python -m driftscope.tides
    python -m driftscope.tides --region sundarbans --days 14
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
    PRESETS,
    Region,
    TIDE_DT_HOURS,
    TIDE_MODEL_NAME,
    TIDES_DIR,
)
from .fetch import output_path as currents_output_path

load_dotenv()
console = Console()


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lat_min,lon_max,lat_max'")
    return tuple(parts)  # type: ignore[return-value]


def output_path(region_name: str, start: str, end: str) -> Path:
    safe = region_name.lower().replace(" ", "_")
    return TIDES_DIR / f"{safe}_{start}_{end}_tides.nc"


def _check_model() -> Path:
    """Verify TIDE_MODEL_DIR is set and exists. Raise with setup hint if not."""
    model_dir = os.getenv("TIDE_MODEL_DIR")
    if not model_dir:
        raise RuntimeError(
            "TIDE_MODEL_DIR not set in .env. To use tides:\n"
            "  1. Register at https://www.tpxo.net (free)\n"
            "  2. Download TPXO9-atlas-v5 (~1 GB, NetCDF format)\n"
            "  3. Set TIDE_MODEL_DIR in .env to the unpacked directory\n"
            "See .env.example for details."
        )
    p = Path(model_dir).expanduser()
    if not p.exists():
        raise RuntimeError(f"TIDE_MODEL_DIR points to non-existent path: {p}")
    return p


def _hourly_times(start_date: str, end_date: str, dt_hours: int) -> np.ndarray:
    """Build a numpy datetime64 array from start to end (inclusive) at dt_hours."""
    s = datetime.fromisoformat(start_date)
    e = datetime.fromisoformat(end_date)
    out = []
    t = s
    while t <= e:
        out.append(np.datetime64(t.strftime("%Y-%m-%dT%H:%M:%S")))
        t += timedelta(hours=dt_hours)
    return np.array(out, dtype="datetime64[s]")


def _predict_grid(
    model_dir: Path, lons: np.ndarray, lats: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Call pyTMD to predict u_tide, v_tide on (time, lat, lon) grid.

    Returns arrays shaped (len(times), len(lats), len(lons)) in m/s.
    Uses pyTMD.compute.tide_currents — this is the v2.x high-level API.
    """
    import pyTMD.compute  # lazy import — heavy startup

    LON, LAT = np.meshgrid(lons, lats)
    flat_x = LON.ravel()
    flat_y = LAT.ravel()
    n_points = flat_x.size
    n_times = times.size

    epoch = np.datetime64("2000-01-01T00:00:00")
    delta_t_seconds = ((times - epoch) / np.timedelta64(1, "s")).astype(float)

    # Cartesian product of (point, time): tile points across each timestamp.
    xx = np.tile(flat_x, n_times)
    yy = np.tile(flat_y, n_times)
    tt = np.repeat(delta_t_seconds, n_points)

    common = dict(
        DIRECTORY=str(model_dir),
        MODEL=TIDE_MODEL_NAME,
        EPOCH=(2000, 1, 1, 0, 0, 0),
        TYPE="drift",
        TIME="UTC",
        EPSG=4326,
        METHOD="spline",
        EXTRAPOLATE=False,
        FILL_VALUE=0.0,
    )
    u_flat = pyTMD.compute.tide_currents(
        x=xx, y=yy, delta_time=tt, COMPONENT="u", **common
    )
    v_flat = pyTMD.compute.tide_currents(
        x=xx, y=yy, delta_time=tt, COMPONENT="v", **common
    )

    # NaN over land (no tide model coverage) → 0 so we don't poison combined currents
    u_flat = np.nan_to_num(np.asarray(u_flat), nan=0.0)
    v_flat = np.nan_to_num(np.asarray(v_flat), nan=0.0)

    u_grid = u_flat.reshape(n_times, len(lats), len(lons))
    v_grid = v_flat.reshape(n_times, len(lats), len(lons))
    return u_grid, v_grid


def fetch_tides(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    currents_nc: Path | None = None,
    force: bool = False,
) -> Path:
    """Predict tidal u/v on the CMEMS spatial grid at hourly time resolution.

    The currents file is read solely to inherit its (lon, lat) grid — we
    don't *use* the current values here. Time is hourly regardless of the
    currents' time cadence so M2/S2/K1/O1 are properly resolved.
    """
    out = output_path(region_name, start_date, end_date)
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    if currents_nc is None:
        currents_nc = currents_output_path(region_name, start_date, end_date)
    currents_nc = Path(currents_nc)
    if not currents_nc.exists():
        raise FileNotFoundError(
            "Need CMEMS currents first to know the grid. Click ⬇️ Fetch currents."
        )

    model_dir = _check_model()

    cur = xr.open_dataset(currents_nc)
    lon_name = "longitude" if "longitude" in cur.coords else "lon"
    lat_name = "latitude" if "latitude" in cur.coords else "lat"
    lons = cur[lon_name].values
    lats = cur[lat_name].values

    times = _hourly_times(start_date, end_date, TIDE_DT_HOURS)

    console.print(
        f"[cyan]↓[/cyan] tides ({TIDE_MODEL_NAME})\n"
        f"   grid: {len(lons)}×{len(lats)}  ·  "
        f"{len(times)} time steps × {TIDE_DT_HOURS}h\n"
        f"   model: {model_dir}"
    )

    u_grid, v_grid = _predict_grid(model_dir, lons, lats, times)

    ds = xr.Dataset(
        data_vars={
            "u_tide": ((("time", lat_name, lon_name)), u_grid,
                       {"long_name": "Barotropic tidal current, eastward",
                        "units": "m s-1"}),
            "v_tide": ((("time", lat_name, lon_name)), v_grid,
                       {"long_name": "Barotropic tidal current, northward",
                        "units": "m s-1"}),
        },
        coords={
            "time": times,
            lat_name: lats,
            lon_name: lons,
        },
    )
    ds.to_netcdf(out)
    console.print(f"[green]✓[/green] saved {out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Predict barotropic tidal currents")
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(PRESETS.keys()))
    p.add_argument("--bbox", type=_parse_bbox, default=None,
                   help="override region: 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--currents", type=Path, default=None,
                   help="CMEMS NetCDF whose grid to use (auto-discovered if omitted)")
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

    fetch_tides(bbox, start, end, region_name=name,
                currents_nc=args.currents, force=args.force)


if __name__ == "__main__":
    main()

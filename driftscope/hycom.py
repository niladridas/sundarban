"""HYCOM ESPC-D-V02 surface-current fetcher.

Independent cross-validation source for CMEMS. Same effective resolution in
longitude (1/12.5° = 9 km) but **2× finer in latitude (1/25° = 4.5 km)**,
which matters for the eastern Bay of Bengal where the drifter validation
showed a missed southward feature.

ESPC-D-V02 is the Naval Oceanographic Office's operational global HYCOM
analysis (1/25° native, served as 0.08° lon × 0.04° lat). Coverage:
2024-08-10 to present, daily means.

Catalog root:
    https://tds.hycom.org/thredds/catalog/datasets/ESPC-D-V02/data/daily_netcdf/

File pattern (per day, ~3GB global):
    US058GCOM-OPSnce.espc-d-031-hycom_fcst_glby008_<YYYYMMDD>12_M0000_<var>.nc

We use OPeNDAP slicing to fetch only surface (depth=0) and a bbox — typically
~10 MB per day instead of 3 GB.

Variable naming alignment with CMEMS:
    HYCOM        CMEMS       what we use
    -----        -----       -----------
    water_u      uo          U component (m/s)
    water_v      vo          V component (m/s)
    lat,lon      latitude,longitude  spatial coords
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console

from .config import CURRENTS_DIR

console = Console()

HYCOM_BASE = (
    "https://tds.hycom.org/thredds/dodsC/datasets/ESPC-D-V02/data/daily_netcdf"
)
HYCOM_FILE_TEMPLATE = (
    "US058GCOM-OPSnce.espc-d-031-hycom_fcst_glby008_{ymd}12_M0000_{var}.nc"
)
HYCOM_TIME_ORIGIN = datetime(2000, 1, 1)


def _hycom_url(date: datetime, var: str) -> str:
    """Build the OPeNDAP URL for a given date and variable ('u3z' or 'v3z')."""
    fname = HYCOM_FILE_TEMPLATE.format(ymd=date.strftime("%Y%m%d"), var=var)
    return f"{HYCOM_BASE}/{date.year}/{fname}"


def _open_hycom_subset(
    url: str, var_name: str,
    bbox: tuple[float, float, float, float],
) -> xr.DataArray:
    """Open a HYCOM file via OPeNDAP, subset to surface and bbox.

    HYCOM's longitudes go 0 → 360. Convert bbox lon (which we use in -180..180
    or 0..360 either way) to the matching range.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # HYCOM uses 0..360 lon — translate negatives if needed.
    lon_min_h = lon_min % 360
    lon_max_h = lon_max % 360
    if lon_max_h < lon_min_h:
        # Bbox crosses 0/360 wrap — not a typical case for BoB, error out.
        raise NotImplementedError("bbox crossing 0/360 wraparound")

    ds = xr.open_dataset(url, decode_times=False)
    da = ds[var_name].isel(time=0, depth=0)  # surface, single time
    da = da.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min_h, lon_max_h),
    )
    # Materialize to memory (this triggers the OPeNDAP byte-range fetch)
    da = da.load()

    # Construct the time stamp from time_origin attr.
    t_origin = ds.attrs.get("time_origin", "")
    if t_origin:
        t = pd.to_datetime(t_origin)
    else:
        # fallback to numeric decode
        hrs = float(ds.time.values[0])
        t = HYCOM_TIME_ORIGIN + timedelta(hours=hrs)
    da = da.expand_dims({"time": [t]})

    # Translate HYCOM lon back to -180..180 if our bbox suggested negative lon
    if lon_min < 0 or lon_max < 0:
        new_lon = ((da.lon + 180) % 360) - 180
        da = da.assign_coords(lon=new_lon).sortby("lon")
    return da


def fetch_hycom_currents(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    region_name: str = "custom",
    force: bool = False,
) -> Path:
    """Fetch HYCOM ESPC-D-V02 surface currents for bbox + date range.

    Output schema mirrors `driftscope.fetch.fetch_currents` (CMEMS):
        Variables: uo (m/s), vo (m/s)
        Coordinates: time, latitude, longitude
    Saved as `data/currents/{region}_{start}_{end}_hycom.nc`.

    Each day is one OPeNDAP fetch per variable (u + v) — a 5×5° BoB bbox at
    0.04° lat × 0.08° lon resolution is roughly 7000 cells × 4 bytes = 28 KB
    per day per variable. 43 days × 2 vars × ~30 KB ≈ 2.5 MB total transfer.
    """
    out = CURRENTS_DIR / f"{region_name.lower().replace(' ', '_')}_{start_date}_{end_date}_hycom.nc"
    if out.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out.name}")
        return out

    sd = pd.Timestamp(start_date).to_pydatetime()
    ed = pd.Timestamp(end_date).to_pydatetime()
    days = pd.date_range(sd, ed, freq="D").to_pydatetime()
    console.print(
        f"[cyan]↓[/cyan] HYCOM ESPC-D-V02 (1/25° lat × 1/12.5° lon)\n"
        f"   bbox: {bbox[0]:.2f},{bbox[1]:.2f} → {bbox[2]:.2f},{bbox[3]:.2f}\n"
        f"   {len(days)} days: {days[0].date()} → {days[-1].date()}"
    )

    u_frames, v_frames = [], []
    for i, day in enumerate(days):
        u_url = _hycom_url(day, "u3z")
        v_url = _hycom_url(day, "v3z")
        try:
            u = _open_hycom_subset(u_url, "water_u", bbox)
            v = _open_hycom_subset(v_url, "water_v", bbox)
            u_frames.append(u)
            v_frames.append(v)
            if (i + 1) % 5 == 0 or i == len(days) - 1:
                console.print(f"   [{i+1}/{len(days)}] {day.date()} ✓")
        except Exception as e:
            console.print(f"   [{i+1}/{len(days)}] {day.date()} [red]✗ {e}[/red]")
            raise

    uo = xr.concat(u_frames, dim="time").rename("uo")
    vo = xr.concat(v_frames, dim="time").rename("vo")
    ds = xr.Dataset({"uo": uo, "vo": vo}).rename({"lat": "latitude", "lon": "longitude"})
    ds.attrs.update({
        "source": "HYCOM ESPC-D-V02 (Navy Operational, 1/25° lat × 1/12.5° lon)",
        "fetched": datetime.utcnow().isoformat(),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    console.print(f"[green]✓[/green] saved {out.name} ({len(days)} days)")
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Fetch HYCOM surface currents")
    p.add_argument("--bbox", required=True,
                   help="lon_min,lat_min,lon_max,lat_max")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--region", default="custom")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    bbox = tuple(float(x) for x in args.bbox.split(","))
    fetch_hycom_currents(bbox, args.start, args.end, args.region, args.force)


if __name__ == "__main__":
    main()

"""Run a Lagrangian particle simulation on a CMEMS NetCDF.

CLI:
    python -m driftscope.simulate
    python -m driftscope.simulate --currents data/currents/foo.nc --particles 1000 --days 14
"""
from __future__ import annotations
import argparse
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console

# parcels v3 + NumPy 2.x: spam from particledata.py using `where=` without `out=`.
# Library-internal, not our bug; remove this filter once parcels patches upstream.
warnings.filterwarnings(
    "ignore",
    message=r".*'where' used without 'out'.*",
    category=UserWarning,
    module=r"parcels\..*",
)

from .config import (
    CURRENTS_DIR,
    PRESETS,
    DEFAULT_REGION,
    TRAJ_DIR,
    SimConfig,
)

console = Console()


def latest_currents_file() -> Path:
    """Most recently modified .nc in data/currents/."""
    files = sorted(CURRENTS_DIR.glob("*.nc"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            "No currents file found. Run `python -m driftscope.fetch` first."
        )
    return files[-1]


def combine_forcings(
    currents_nc: Path,
    stokes_nc: Path | None = None,
    tides_nc: Path | None = None,
    winds_nc: Path | None = None,
    stokes_scale: float = 1.0,
    windage_coeff: float = 0.03,
) -> Path:
    """Build a single U/V NetCDF that sums the requested forcings.

    - Currents (always): CMEMS at native (typically daily) cadence.
    - Stokes (optional): ERA5-derived; regridded to currents grid, scaled.
    - Tides (optional): pyTMD predictions at hourly cadence — when included,
      the output is at the *tides* time grid (hourly), with daily currents
      forward-filled to each hour. This preserves M2/S2 oscillations that
      would otherwise be averaged away.
    - Winds (optional): ERA5 10m wind, multiplied by `windage_coeff` (typical
      0.01–0.04) and added to U/V. Captures direct wind drag on the
      air-exposed fraction of floating objects.

    Cache filename encodes which forcings are active so different combos
    don't collide. Re-runs are cheap: regenerated only if any input is newer.
    """
    tag = []
    if stokes_nc is not None:
        tag.append(f"stokes{stokes_scale:.1f}")
    if tides_nc is not None:
        tag.append("tides")
    if winds_nc is not None:
        tag.append(f"wind{windage_coeff*100:.1f}pct")
    suffix = ("__" + "_".join(tag)) if tag else "__currents_only"
    out = currents_nc.with_name(f"{currents_nc.stem}{suffix}.nc")

    inputs = [currents_nc] + [p for p in (stokes_nc, tides_nc, winds_nc) if p is not None]
    if out.exists():
        out_mtime = out.stat().st_mtime
        if all(out_mtime >= p.stat().st_mtime for p in inputs):
            return out

    cur = xr.open_dataset(currents_nc)
    cur_lon = "longitude" if "longitude" in cur.coords else "lon"
    cur_lat = "latitude" if "latitude" in cur.coords else "lat"

    # Pick the time grid: tides (hourly) wins if present, else currents (daily).
    if tides_nc is not None:
        tid = xr.open_dataset(tides_nc)
        time_grid = tid["time"]
        cur_resampled = cur.reindex(time=time_grid, method="nearest")
    else:
        time_grid = cur["time"]
        cur_resampled = cur

    uo = cur_resampled["uo"].copy()
    vo = cur_resampled["vo"].copy()

    if stokes_nc is not None:
        wav = xr.open_dataset(stokes_nc)
        if "valid_time" in wav.coords and "time" not in wav.coords:
            wav = wav.rename({"valid_time": "time"})
        wav_lon = "longitude" if "longitude" in wav.coords else "lon"
        wav_lat = "latitude" if "latitude" in wav.coords else "lat"
        ust = wav["ust"].interp(
            {wav_lon: cur[cur_lon], wav_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0) * stokes_scale
        vst = wav["vst"].interp(
            {wav_lon: cur[cur_lon], wav_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0) * stokes_scale
        uo = uo + ust
        vo = vo + vst

    if tides_nc is not None:
        tid_lon = "longitude" if "longitude" in tid.coords else "lon"
        tid_lat = "latitude" if "latitude" in tid.coords else "lat"
        u_tide = tid["u_tide"].interp(
            {tid_lon: cur[cur_lon], tid_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0)
        v_tide = tid["v_tide"].interp(
            {tid_lon: cur[cur_lon], tid_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0)
        uo = uo + u_tide
        vo = vo + v_tide

    if winds_nc is not None:
        wnd = xr.open_dataset(winds_nc)
        if "valid_time" in wnd.coords and "time" not in wnd.coords:
            wnd = wnd.rename({"valid_time": "time"})
        wnd_lon = "longitude" if "longitude" in wnd.coords else "lon"
        wnd_lat = "latitude" if "latitude" in wnd.coords else "lat"
        u10 = wnd["u10"].interp(
            {wnd_lon: cur[cur_lon], wnd_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0) * windage_coeff
        v10 = wnd["v10"].interp(
            {wnd_lon: cur[cur_lon], wnd_lat: cur[cur_lat], "time": time_grid},
        ).fillna(0.0) * windage_coeff
        uo = uo + u10
        vo = vo + v10

    combined = cur_resampled.copy()
    combined["uo"] = uo
    combined["vo"] = vo
    combined.to_netcdf(out)
    return out


# Backward-compat alias — older code/tests still call this name.
def combine_currents_and_stokes(currents_nc, stokes_nc, scale=1.0):
    return combine_forcings(currents_nc, stokes_nc=stokes_nc, stokes_scale=scale)


def seed_disk(
    lon0: float, lat0: float, radius_deg: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample n points inside a disk centered at (lon0, lat0)."""
    r = radius_deg * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    return lon0 + r * np.cos(theta), lat0 + r * np.sin(theta)


def run_simulation(
    currents_nc: Path,
    cfg: SimConfig,
    out_path: Path | None = None,
    seed: int = 42,
    stokes_nc: Path | None = None,
    tides_nc: Path | None = None,
    winds_nc: Path | None = None,
) -> Path:
    """Advect `cfg.n_particles` particles for `cfg.runtime_days` days.

    Optional forcings are added at the data layer before advection:
      - cfg.include_stokes + stokes_nc → ERA5-derived surface Stokes drift
      - cfg.include_tides + tides_nc   → pyTMD barotropic tidal currents
      - cfg.include_winds + winds_nc   → ERA5 10m wind × cfg.windage_coeff

    Returns the path to the trajectory file (Zarr).
    """
    # Lazy import — parcels has heavy startup
    from parcels import (
        FieldSet,
        ParticleSet,
        JITParticle,
        AdvectionRK4,
        StatusCode,
        Variable,
    )
    from .kernels import Settling, stokes_settling_velocity

    use_stokes = cfg.include_stokes and stokes_nc is not None
    use_tides = cfg.include_tides and tides_nc is not None
    use_winds = cfg.include_winds and winds_nc is not None
    use_settling = cfg.include_settling
    if use_stokes or use_tides or use_winds:
        active = []
        if use_stokes:
            active.append(f"Stokes×{cfg.stokes_scale:.1f}")
        if use_tides:
            active.append("tides")
        if use_winds:
            active.append(f"wind×{cfg.windage_coeff*100:.1f}%")
        console.print(f"[cyan]·[/cyan] forcings: currents + {' + '.join(active)}")
        currents_nc = combine_forcings(
            currents_nc,
            stokes_nc=stokes_nc if use_stokes else None,
            tides_nc=tides_nc if use_tides else None,
            winds_nc=winds_nc if use_winds else None,
            stokes_scale=cfg.stokes_scale,
            windage_coeff=cfg.windage_coeff,
        )

    # Settling lives in a kernel (per-particle vertical velocity, not a field).
    if use_settling:
        w_s = stokes_settling_velocity(
            cfg.settling_diameter_um, cfg.settling_density_kg_m3
        )
        console.print(
            f"[cyan]·[/cyan] settling: {cfg.settling_diameter_um:.1f}µm "
            f"@ {cfg.settling_density_kg_m3:.0f} kg/m³ → "
            f"w_s = {w_s*1000:.4f} mm/s ({w_s*86400:.2f} m/day)"
        )

        class SedimentParticle(JITParticle):
            w_settle = Variable("w_settle", dtype=np.float32, initial=w_s)
        ParticleClass = SedimentParticle
    else:
        ParticleClass = JITParticle

    console.print(f"[cyan]·[/cyan] loading currents: {currents_nc.name}")
    ds = xr.open_dataset(currents_nc)

    # CMEMS uses 'longitude'/'latitude'/'time'; some products use 'lon'/'lat'.
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"

    filenames = {
        "U": str(currents_nc),
        "V": str(currents_nc),
    }
    variables = {"U": "uo", "V": "vo"}
    dimensions = {
        "U": {"lon": lon_name, "lat": lat_name, "time": "time"},
        "V": {"lon": lon_name, "lat": lat_name, "time": "time"},
    }

    fieldset = FieldSet.from_netcdf(
        filenames, variables, dimensions, allow_time_extrapolation=True
    )

    # ── Seeding ──────────────────────────────────────────────────────────────
    lons = ds[lon_name].values
    lats = ds[lat_name].values
    rng = np.random.default_rng(seed)

    if cfg.seed_mode == "bbox":
        # Basin fill: oversample uniformly across the data bbox; the land/NaN
        # filter below retains only ocean cells, giving a coastline-shaped spread.
        oversample = 5
        n_cand = cfg.n_particles * oversample
        plon = rng.uniform(float(lons.min()), float(lons.max()), n_cand)
        plat = rng.uniform(float(lats.min()), float(lats.max()), n_cand)
    else:
        seed_lon = cfg.seed_lon if cfg.seed_lon is not None else float(lons.mean())
        seed_lat = cfg.seed_lat if cfg.seed_lat is not None else float(lats.mean())
        plon, plat = seed_disk(
            seed_lon, seed_lat, cfg.seed_radius_deg, cfg.n_particles, rng
        )

    # Filter out seeds that fall on NaN (land) at t=0
    u0 = ds.uo.isel(time=0).squeeze()
    valid = []
    for i, (x, y) in enumerate(zip(plon, plat)):
        try:
            v = float(u0.sel({lon_name: x, lat_name: y}, method="nearest"))
            if np.isfinite(v):
                valid.append(i)
        except Exception:
            pass
    plon, plat = plon[valid], plat[valid]

    # Trim oversampled bbox candidates down to the requested count.
    if cfg.seed_mode == "bbox" and len(plon) > cfg.n_particles:
        plon = plon[: cfg.n_particles]
        plat = plat[: cfg.n_particles]

    if cfg.seed_mode == "bbox":
        console.print(
            f"[cyan]·[/cyan] basin-fill: seeded {len(plon)}/{cfg.n_particles} "
            f"ocean particles across bbox"
        )
    else:
        console.print(
            f"[cyan]·[/cyan] seeded {len(plon)}/{cfg.n_particles} particles "
            f"at ({seed_lon:.3f}, {seed_lat:.3f}) r={cfg.seed_radius_deg}°"
        )

    if len(plon) == 0:
        raise RuntimeError(
            "All seed points landed on land/NaN. For 'disk' mode, move "
            "seed_lon/seed_lat offshore or increase seed_radius_deg. For 'bbox' "
            "mode, check that the bbox actually contains ocean."
        )

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=ParticleClass,
        lon=plon,
        lat=plat,
        time=ds.time.values[0],
    )

    # ── Output ───────────────────────────────────────────────────────────────
    if out_path is None:
        stem = currents_nc.stem
        if use_settling:
            stem = f"{stem}__settle{cfg.settling_diameter_um:g}um"
        out_path = TRAJ_DIR / f"{stem}_traj.zarr"

    output = pset.ParticleFile(
        name=str(out_path),
        outputdt=timedelta(minutes=cfg.output_minutes),
    )

    # ── Run ──────────────────────────────────────────────────────────────────
    console.print(
        f"[cyan]→[/cyan] running {cfg.runtime_days}d, "
        f"dt={cfg.dt_minutes}min, save every {cfg.output_minutes}min"
    )

    def DeleteOOB(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    kernels = [AdvectionRK4]
    if use_settling:
        kernels.append(Settling)
    kernels.append(DeleteOOB)

    pset.execute(
        kernels,
        runtime=timedelta(days=cfg.runtime_days),
        dt=timedelta(minutes=cfg.dt_minutes),
        output_file=output,
    )

    console.print(f"[green]✓[/green] trajectories: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run a particle simulation")
    p.add_argument("--currents", type=Path, default=None,
                   help="CMEMS NetCDF (default: latest in data/currents/)")
    p.add_argument("--particles", type=int, default=500)
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--seed-lon", type=float, default=None)
    p.add_argument("--seed-lat", type=float, default=None)
    p.add_argument("--seed-radius", type=float, default=0.15)
    p.add_argument("--seed-mode", default="disk", choices=["disk", "bbox"],
                   help="'disk' = small release at seed_lon/lat; "
                        "'bbox' = basin fill across all ocean cells")
    p.add_argument("--stokes", type=Path, default=None,
                   help="ERA5 Stokes-drift NetCDF (from `python -m driftscope.stokes`)")
    p.add_argument("--stokes-scale", type=float, default=1.0,
                   help="multiplier on Stokes magnitude (1.0 baseline, ~8.0 = OpenDrift)")
    p.add_argument("--tides", type=Path, default=None,
                   help="pyTMD tide NetCDF (from `python -m driftscope.tides`)")
    p.add_argument("--winds", type=Path, default=None,
                   help="ERA5 wind NetCDF (from `python -m driftscope.winds`)")
    p.add_argument("--windage", type=float, default=0.03,
                   help="windage coefficient (default 0.03 = 3%%)")
    p.add_argument("--settle", action="store_true",
                   help="enable Stokes settling (Module 5)")
    p.add_argument("--settle-diameter-um", type=float, default=10.0,
                   help="particle diameter in µm (default 10 = silt-class)")
    p.add_argument("--settle-density", type=float, default=2650.0,
                   help="particle density kg/m³ (default 2650 = quartz)")
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(PRESETS.keys()),
                   help="if seed-lon/lat omitted, use this region's center")
    args = p.parse_args()

    currents = args.currents or latest_currents_file()

    if args.seed_lon is None or args.seed_lat is None:
        r = PRESETS[args.region]
        sl, slat = r.default_seed
    else:
        sl, slat = args.seed_lon, args.seed_lat

    cfg = SimConfig(
        n_particles=args.particles,
        runtime_days=args.days,
        seed_lon=sl,
        seed_lat=slat,
        seed_radius_deg=args.seed_radius,
        seed_mode=args.seed_mode,
        include_stokes=args.stokes is not None,
        stokes_scale=args.stokes_scale,
        include_tides=args.tides is not None,
        include_winds=args.winds is not None,
        windage_coeff=args.windage,
        include_settling=args.settle,
        settling_diameter_um=args.settle_diameter_um,
        settling_density_kg_m3=args.settle_density,
    )

    run_simulation(
        currents, cfg,
        stokes_nc=args.stokes, tides_nc=args.tides, winds_nc=args.winds,
    )


if __name__ == "__main__":
    main()

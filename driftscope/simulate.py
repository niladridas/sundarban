"""Run a Lagrangian particle simulation on a CMEMS NetCDF.

CLI:
    python -m driftscope.simulate
    python -m driftscope.simulate --currents data/currents/foo.nc --particles 1000 --days 14
"""
from __future__ import annotations
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console

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


def combine_currents_and_stokes(
    currents_nc: Path, stokes_nc: Path, scale: float = 1.0
) -> Path:
    """Regrid Stokes drift onto the CMEMS currents grid and add to U/V.

    `scale` multiplies the derived Stokes magnitude before it's added (Phillips
    prefactor varies 1×–8× across literature; this exposes the choice).

    The cache filename includes the scale so different scales don't collide.
    Re-runs are cheap: regenerated only if either input is newer.
    """
    out = currents_nc.with_name(f"{currents_nc.stem}__stokes_s{scale:.1f}.nc")
    if out.exists():
        out_mtime = out.stat().st_mtime
        if out_mtime >= currents_nc.stat().st_mtime and out_mtime >= stokes_nc.stat().st_mtime:
            return out

    cur = xr.open_dataset(currents_nc)
    wav = xr.open_dataset(stokes_nc)

    # Newer ERA5 downloads use `valid_time`; older use `time`.
    if "valid_time" in wav.coords and "time" not in wav.coords:
        wav = wav.rename({"valid_time": "time"})

    cur_lon = "longitude" if "longitude" in cur.coords else "lon"
    cur_lat = "latitude" if "latitude" in cur.coords else "lat"
    wav_lon = "longitude" if "longitude" in wav.coords else "lon"
    wav_lat = "latitude" if "latitude" in wav.coords else "lat"

    # Interpolate Stokes onto the currents (lon, lat, time) grid.
    # ERA5 is coarser than CMEMS, so this is upsampling — purely smoothing.
    interp_kwargs = {
        wav_lon: cur[cur_lon],
        wav_lat: cur[cur_lat],
        "time": cur["time"],
    }
    ust = wav["ust"].interp(**interp_kwargs).fillna(0.0) * scale
    vst = wav["vst"].interp(**interp_kwargs).fillna(0.0) * scale

    combined = cur.copy()
    # Broadcast Stokes (no depth dim) over the currents' depth dim if present.
    combined["uo"] = cur["uo"] + ust
    combined["vo"] = cur["vo"] + vst
    combined.to_netcdf(out)
    return out


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
) -> Path:
    """Advect `cfg.n_particles` particles for `cfg.runtime_days` days.

    If `cfg.include_stokes` is True and `stokes_nc` is provided, surface Stokes
    drift is added to the current velocity at the data layer before advection.

    Returns the path to the trajectory file (Zarr).
    """
    # Lazy import — parcels has heavy startup
    from parcels import (
        FieldSet,
        ParticleSet,
        JITParticle,
        AdvectionRK4,
        StatusCode,
    )

    if cfg.include_stokes and stokes_nc is not None:
        console.print(
            f"[cyan]·[/cyan] adding Stokes drift from {stokes_nc.name} "
            f"(scale={cfg.stokes_scale:.1f})"
        )
        currents_nc = combine_currents_and_stokes(
            currents_nc, stokes_nc, scale=cfg.stokes_scale
        )

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
        pclass=JITParticle,
        lon=plon,
        lat=plat,
        time=ds.time.values[0],
    )

    # ── Output ───────────────────────────────────────────────────────────────
    if out_path is None:
        stem = currents_nc.stem
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

    pset.execute(
        [AdvectionRK4, DeleteOOB],
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
    )

    run_simulation(currents, cfg, stokes_nc=args.stokes)


if __name__ == "__main__":
    main()

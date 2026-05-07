"""GBM river-mouth seasonal dispersion simulation + animation.

Releases ~2400 particles distributed across 12 GBM distributaries weighted
by approximate discharge. Advects 30 days under HYCOM currents + Brownian
diffusion. Renders an MP4 with cartopy basemap + per-river coloring.

Configurable for 4 seasonal scenarios (pre-monsoon, peak monsoon,
post-monsoon, dry) by changing SEASON below or via the script CLI.

The aim is **visual storytelling** — particles emerging from each river
mouth, mixing in the coastal zone, sweeping out into the Bay of Bengal
under that season's currents.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console

from driftscope.config import SimConfig, CURRENTS_DIR, TRAJ_DIR, DATA
from driftscope.simulate import run_simulation
from driftscope.rivers import GBM_RIVERS, sample_river_releases
from driftscope.animate import animate_trajectories

console = Console()

# ── Seasonal scenarios ───────────────────────────────────────────────────────
# Each season picks a 30-day HYCOM window. Picked years where HYCOM has
# coverage (≥ 2024-08-10).
SEASONS = {
    "premonsoon":  {
        "start": "2025-04-20", "end": "2025-05-20",
        "label": "Pre-monsoon (Apr–May 2025)",
        "hycom_file": "gbm_premonsoon_2025-04-20_2025-06-01_hycom.nc",
    },
    "monsoon": {
        "start": "2025-08-01", "end": "2025-08-31",
        "label": "Peak monsoon (Aug 2025)",
        "hycom_file": "gbm_monsoon_2025-08-01_2025-08-31_hycom.nc",
    },
    "postmonsoon": {
        "start": "2025-10-15", "end": "2025-11-14",
        "label": "Post-monsoon (Oct–Nov 2025)",
        "hycom_file": "gbm_postmonsoon_2025-10-15_2025-11-14_hycom.nc",
    },
    "dry": {
        "start": "2025-01-15", "end": "2025-02-14",
        "label": "Dry season (Jan–Feb 2025)",
        "hycom_file": "gbm_dry_2025-01-15_2025-02-14_hycom.nc",
    },
}

GBM_BBOX = (87.5, 19.0, 92.5, 23.0)
N_PARTICLES = 2400
RUNTIME_DAYS = 25                   # leave 5d slack vs 30d HYCOM window
RANDOM_SEED = 42
BROWNIAN_KH_M2_S = 80.0             # slightly conservative for visual clarity


def run_season(season: str, render_mp4: bool = True) -> tuple[Path, Path | None]:
    cfg_season = SEASONS[season]
    hycom_nc = CURRENTS_DIR / cfg_season["hycom_file"]
    if not hycom_nc.exists():
        raise FileNotFoundError(
            f"HYCOM file missing: {hycom_nc}\n"
            f"Run: python -m driftscope.hycom --bbox 87.5,19,92.5,23 "
            f"--start {cfg_season['start']} --end {cfg_season['end']} "
            f"--region gbm_{season}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    plon, plat, ridx = sample_river_releases(N_PARTICLES, rng)

    # Land-mask filter here (rather than letting run_simulation do it) so
    # river_idx stays aligned with the surviving particles. River mouths often
    # fall on cells HYCOM marks as land; oversample upstream and re-roll until
    # we hit the requested count.
    cur = xr.open_dataset(hycom_nc)
    u0 = cur.uo.isel(time=0).squeeze()
    keep = []
    for x, y in zip(plon, plat):
        try:
            v = float(u0.sel(longitude=x, latitude=y, method="nearest"))
            keep.append(np.isfinite(v))
        except Exception:
            keep.append(False)
    keep = np.array(keep)
    plon, plat, ridx = plon[keep], plat[keep], ridx[keep]
    console.print(
        f"[cyan]·[/cyan] {len(plon)}/{N_PARTICLES} particles in ocean cells "
        f"across {len(GBM_RIVERS)} river mouths"
    )

    cfg = SimConfig(
        n_particles=len(plon),
        runtime_days=RUNTIME_DAYS,
        dt_minutes=30,
        output_minutes=180,
        # Currents only + Brownian for now — keep it visually clean. Can
        # turn on tides/winds/Stokes for production runs once visuals are tuned.
        include_brownian=True, brownian_kh=BROWNIAN_KH_M2_S,
    )

    traj_out = TRAJ_DIR / f"gbm_{season}_{RUNTIME_DAYS}d.zarr"
    if traj_out.exists():
        import shutil
        shutil.rmtree(traj_out)
    run_simulation(
        hycom_nc, cfg,
        out_path=traj_out, seed=RANDOM_SEED,
        seed_lons=plon, seed_lats=plat,
    )

    # Save river_idx alongside the traj so the animator can color by source
    np.save(traj_out.parent / f"{traj_out.stem}_river_idx.npy", ridx)

    if not render_mp4:
        return traj_out, None

    mp4_out = DATA / "animations" / f"gbm_{season}.mp4"
    animate_trajectories(
        traj_zarr=traj_out,
        out_path=mp4_out,
        bbox=GBM_BBOX,
        river_idx=np.load(traj_out.parent / f"{traj_out.stem}_river_idx.npy"),
        river_names=[r.name for r in GBM_RIVERS],
        fps=15, tail_steps=8, point_size=10.0,
        title_prefix=cfg_season["label"],
    )
    return traj_out, mp4_out


def main():
    p = argparse.ArgumentParser(description="Run a seasonal GBM dispersion + animation")
    p.add_argument("season", choices=list(SEASONS.keys()))
    p.add_argument("--no-mp4", action="store_true")
    args = p.parse_args()
    run_season(args.season, render_mp4=not args.no_mp4)


if __name__ == "__main__":
    main()

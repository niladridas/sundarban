"""Coupled DriftScope simulation: SPM-weighted release + all four forcings + settling.

This is the payoff scenario the project was built toward. Steps:

  1. Load the Module-7 SPM map (CBR retrieval over Sundarbans).
  2. Sample N release points weighted by SPM concentration — particles start
     where there's actually sediment in the water.
  3. Advect for the configured runtime under:
       currents (CMEMS) + tides (TPXO9) + Stokes drift (ERA5 waves)
       + windage (ERA5 wind) + Stokes settling (Module 5).
  4. Save trajectories and produce a dispersion plot.

The SPM map is from 2024-03-08 (dry season) but the forcings are 2026-04-17
onward. We use the SPM map purely as a *spatial release pattern* — where
sediment was concentrated in a representative dry-season scene — not as
a time-aligned tracer concentration. For temporally-aligned runs we'd
need a Landsat scene from the simulation period.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console

from driftscope.config import SimConfig, CURRENTS_DIR, TIDES_DIR, WAVES_DIR, WINDS_DIR, SPM_DIR, TRAJ_DIR
from driftscope.simulate import run_simulation

console = Console()

# ── Inputs ───────────────────────────────────────────────────────────────────
SPM_NC = SPM_DIR / "sundarbans_spm_2024-03-08.nc"
CURRENTS_NC = CURRENTS_DIR / "sundarbans_delta_2026-04-17_2026-05-01.nc"
TIDES_NC = TIDES_DIR / "sundarbans_delta_2026-04-17_2026-05-01_tides.nc"
WAVES_NC = WAVES_DIR / "sundarbans_delta_2026-04-17_2026-05-01_waves.nc"
WINDS_NC = WINDS_DIR / "sundarbans_delta_2026-04-17_2026-05-01_winds.nc"

# ── Run parameters ───────────────────────────────────────────────────────────
N_PARTICLES = 2000
RUNTIME_DAYS = 10
RANDOM_SEED = 42
SETTLE_DIAMETER_UM = 10.0   # silt-class quartz; matches Module 5 default
SETTLE_DENSITY = 2650.0
WINDAGE_COEFF = 0.02         # 2% — typical for sediment-laden water (lower than oil)


def sample_spm_weighted_seeds(
    spm_nc: Path,
    currents_nc: Path,
    n: int,
    rng: np.random.Generator,
    min_spm_mg_l: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (lon, lat) release points weighted by SPM concentration.

    Constraints:
      - Each sampled pixel must have valid SPM > min_spm_mg_l.
      - Each sample must fall inside the currents ocean mask (uo finite at t=0)
        — otherwise the run_simulation land filter will discard it.

    Returns plon, plat as 1D arrays of length ≤ n (post-mask).
    """
    spm = xr.open_dataset(spm_nc).load().spm_mg_l   # (lat, lon)
    cur = xr.open_dataset(currents_nc).load()
    u0 = cur.uo.isel(time=0).squeeze()              # (lat, lon)
    cur_lon_name = "longitude" if "longitude" in cur.coords else "lon"
    cur_lat_name = "latitude" if "latitude" in cur.coords else "lat"

    # Build the per-pixel weight grid: SPM where finite & above threshold, else 0.
    w = spm.values.astype(float)
    valid = np.isfinite(w) & (w > min_spm_mg_l)
    w = np.where(valid, w, 0.0)

    if w.sum() == 0:
        raise RuntimeError(
            f"No SPM pixels above {min_spm_mg_l} mg/L in {spm_nc.name}"
        )

    # Flatten + normalize → multinomial sampling
    flat_w = w.ravel() / w.sum()
    n_pix = len(flat_w)
    # Oversample to absorb the currents-grid land-mask attrition.
    n_cand = min(n * 4, valid.sum())
    idx = rng.choice(n_pix, size=n_cand, replace=True, p=flat_w)
    lat_idx, lon_idx = np.unravel_index(idx, w.shape)
    plat = spm.latitude.values[lat_idx]
    plon = spm.longitude.values[lon_idx]

    # Filter to currents-grid ocean cells (finite uo at t=0)
    keep = []
    for x, y in zip(plon, plat):
        try:
            v = float(u0.sel({cur_lon_name: x, cur_lat_name: y}, method="nearest"))
            if np.isfinite(v):
                keep.append(True)
            else:
                keep.append(False)
        except Exception:
            keep.append(False)
    keep = np.array(keep)
    plon, plat = plon[keep], plat[keep]

    if len(plon) > n:
        choice = rng.choice(len(plon), size=n, replace=False)
        plon, plat = plon[choice], plat[choice]

    console.print(
        f"[green]✓[/green] SPM-weighted release: {len(plon)}/{n} particles "
        f"(median SPM at sampled pixels: "
        f"{float(np.median(w[lat_idx[keep], lon_idx[keep]])):.1f} mg/L)"
    )
    return plon, plat


def main() -> Path:
    rng = np.random.default_rng(RANDOM_SEED)

    for p, name in [(SPM_NC, "SPM map"), (CURRENTS_NC, "currents"),
                    (TIDES_NC, "tides"), (WAVES_NC, "waves"), (WINDS_NC, "winds")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} input missing: {p}")

    console.print(f"[cyan]·[/cyan] sampling SPM-weighted release pattern")
    plon, plat = sample_spm_weighted_seeds(
        SPM_NC, CURRENTS_NC, n=N_PARTICLES, rng=rng,
    )

    cfg = SimConfig(
        n_particles=len(plon),
        runtime_days=RUNTIME_DAYS,
        dt_minutes=30,
        output_minutes=180,
        # seed_mode/seed_lon/seed_lat all bypassed by seed_lons/seed_lats below
        include_stokes=True, stokes_scale=1.0,
        include_tides=True,
        include_winds=True, windage_coeff=WINDAGE_COEFF,
        include_settling=True,
        settling_diameter_um=SETTLE_DIAMETER_UM,
        settling_density_kg_m3=SETTLE_DENSITY,
    )

    out = TRAJ_DIR / f"coupled_spm_seeded_{RUNTIME_DAYS}d.zarr"
    return run_simulation(
        CURRENTS_NC, cfg,
        out_path=out, seed=RANDOM_SEED,
        stokes_nc=WAVES_NC, tides_nc=TIDES_NC, winds_nc=WINDS_NC,
        seed_lons=plon, seed_lats=plat,
    )


if __name__ == "__main__":
    main()

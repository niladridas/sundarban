"""GDP drifter ↔ DriftScope skill comparison.

The most credible single image you can produce: take a real surface drifter,
release a single simulated particle at the drifter's first observed position
and time, advect it with the same currents, and overlay both tracks.

Picks GDP drifter 300534064293930 — 42-day track in the Bay of Bengal
2025-04-20 → 2025-06-01, lat 12–14.5°N, lon 89.3–92.4°E. Net displacement
374 km, total path 867 km. The drifter is drogued at 15m, so windage and
Stokes drift do NOT apply (they're surface-only effects); we test the
*currents* alone, which is the right baseline.

Skill metrics:
  - Separation distance s(t) = haversine(model_t, drifter_t)
  - Liu-Weisberg skill score: 1 - mean(s)/mean(d_drifter), where d_drifter
    is cumulative distance traveled by the drifter. 1.0 = perfect, 0.0 = no
    skill, negative = anti-skill. Standard in Lagrangian-validation papers.
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from rich.console import Console

from driftscope.config import CURRENTS_DIR, DRIFTERS_DIR, SimConfig, TRAJ_DIR
from driftscope.simulate import run_simulation
from driftscope.viz import load_trajectories, traj_to_dataframe

console = Console()

# ── Inputs ───────────────────────────────────────────────────────────────────
DRIFTER_CSV = DRIFTERS_DIR / "gdp_300534064293930_2025-04-20_2025-06-01.csv"
CURRENTS_NC = CURRENTS_DIR / "custom_2025-04-20_2025-06-01.nc"
TRAJ_OUT = TRAJ_DIR / "drifter_validation_300534064293930.zarr"
OUT_PNG = Path(__file__).resolve().parent.parent / "data" / "drifter_validation.png"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Load drifter ─────────────────────────────────────────────────────────────
drf = pd.read_csv(DRIFTER_CSV, skiprows=[1])
drf["time"] = pd.to_datetime(drf["time"]).dt.tz_localize(None)  # match parcels' tz-naive output
drf = drf.sort_values("time").reset_index(drop=True)
drf = drf.rename(columns={"latitude": "lat", "longitude": "lon"})
console.print(
    f"[cyan]·[/cyan] drifter {drf.ID.iloc[0]}: {len(drf)} positions over "
    f"{(drf.time.max() - drf.time.min()).total_seconds()/86400:.1f} days"
)

# ── Configure simulation matched to drifter ──────────────────────────────────
n_days = int(np.ceil((drf.time.max() - drf.time.min()).total_seconds() / 86400))
cfg = SimConfig(
    n_particles=1,            # single particle to match the single drifter
    runtime_days=n_days,
    dt_minutes=30,
    output_minutes=60,        # 1h save cadence — finer than drifter's 6h
    # No tides/winds/Stokes/settling: drifter is drogued at 15m. We're testing
    # the CMEMS surface-current advection alone.
    include_stokes=False, include_tides=False,
    include_winds=False, include_settling=False,
)

# Currents NetCDF starts at midnight 2025-04-20; drifter starts at midnight 2025-04-20.
# Just verify alignment.
ds = xr.open_dataset(CURRENTS_NC)
console.print(
    f"[cyan]·[/cyan] currents: {str(ds.time.values[0])[:19]} → "
    f"{str(ds.time.values[-1])[:19]} ({len(ds.time)} steps)"
)
console.print(
    f"[cyan]·[/cyan] drifter:  {drf.time.iloc[0]} → {drf.time.iloc[-1]}"
)

# ── Run single-particle simulation ───────────────────────────────────────────
seed_lon = float(drf.lon.iloc[0])
seed_lat = float(drf.lat.iloc[0])
console.print(
    f"[cyan]→[/cyan] seeding 1 particle at "
    f"({seed_lat:.3f}°N, {seed_lon:.3f}°E) — drifter's t0 position"
)

if TRAJ_OUT.exists():
    import shutil
    shutil.rmtree(TRAJ_OUT)

run_simulation(
    CURRENTS_NC, cfg,
    out_path=TRAJ_OUT,
    seed=42,
    seed_lons=np.array([seed_lon]),
    seed_lats=np.array([seed_lat]),
)

# ── Load simulated trajectory ────────────────────────────────────────────────
sim_ds = load_trajectories(TRAJ_OUT)
sim_df = traj_to_dataframe(sim_ds).sort_values("time").reset_index(drop=True)
console.print(f"[cyan]·[/cyan] simulated track: {len(sim_df)} positions")

# ── Co-locate in time: nearest-neighbor each drifter time to a sim time ──────
def nearest_sim_position(t: pd.Timestamp) -> tuple[float, float]:
    idx = (sim_df["time"] - t).abs().idxmin()
    return sim_df.loc[idx, "lon"], sim_df.loc[idx, "lat"]

paired = []
for _, r in drf.iterrows():
    sl, sa = nearest_sim_position(r["time"])
    paired.append({
        "time": r["time"],
        "drifter_lat": r["lat"], "drifter_lon": r["lon"],
        "sim_lat": sa, "sim_lon": sl,
    })
pf = pd.DataFrame(paired)
pf["sep_km"] = haversine_km(pf.drifter_lat, pf.drifter_lon, pf.sim_lat, pf.sim_lon)
pf["t_days"] = (pf["time"] - pf["time"].iloc[0]).dt.total_seconds() / 86400.0

# Cumulative distance the drifter actually traveled (denominator of LW skill)
drf_lat, drf_lon = pf.drifter_lat.values, pf.drifter_lon.values
seg = haversine_km(drf_lat[:-1], drf_lon[:-1], drf_lat[1:], drf_lon[1:])
cum_drifter_km = np.concatenate([[0], np.cumsum(seg)])

# Liu-Weisberg skill score — clamp to non-negative for reporting
# (negative = anti-skill, often reported as 0).
mean_sep = float(pf["sep_km"].mean())
mean_cum = float(np.mean(cum_drifter_km[1:]))   # exclude t=0
lw_skill = 1.0 - mean_sep / mean_cum if mean_cum > 0 else float("nan")

console.print()
console.print(f"[bold]Skill metrics[/bold] over {pf.t_days.iloc[-1]:.1f} days:")
console.print(f"  separation  median: {pf.sep_km.median():.1f} km")
console.print(f"  separation     P90: {pf.sep_km.quantile(0.9):.1f} km")
console.print(f"  separation     max: {pf.sep_km.max():.1f} km")
console.print(f"  drifter total path: {cum_drifter_km[-1]:.1f} km")
console.print(f"  drifter end-to-end: "
              f"{haversine_km(drf_lat[0], drf_lon[0], drf_lat[-1], drf_lon[-1]):.1f} km")
console.print(f"  Liu-Weisberg skill: [bold]{lw_skill:.3f}[/bold]")
console.print(f"  (1.0 = perfect, 0 = no skill, < 0 = anti-skill)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(drf.lon, drf.lat, "-", lw=1.4, color="black", label="GDP drifter (real)")
ax.scatter(drf.lon.iloc[0], drf.lat.iloc[0], marker="o", s=80,
           facecolor="lime", edgecolor="black", zorder=5, label="release (t=0)")
ax.scatter(drf.lon.iloc[-1], drf.lat.iloc[-1], marker="X", s=80,
           facecolor="black", edgecolor="white", zorder=5, label=f"drifter end (t={n_days}d)")
ax.plot(sim_df.lon, sim_df.lat, "-", lw=1.4, color="crimson", label="DriftScope sim")
ax.scatter(sim_df.lon.iloc[-1], sim_df.lat.iloc[-1], marker="X", s=80,
           facecolor="crimson", edgecolor="white", zorder=5, label=f"sim end (t={n_days}d)")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"GDP drifter {drf.ID.iloc[0]} vs DriftScope (currents only)")
ax.legend(loc="best")
ax.set_aspect("equal")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(pf.t_days, pf.sep_km, "-", color="navy", lw=1.4, label="separation")
ax.axhline(pf.sep_km.median(), color="red", ls=":", lw=1, label=f"median {pf.sep_km.median():.0f} km")
ax.set_xlabel("Days since release")
ax.set_ylabel("Separation (km)")
ax.set_title(f"Sim ↔ drifter separation\nLiu-Weisberg skill = {lw_skill:.3f}")
ax.legend(loc="best")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
console.print(f"[green]✓[/green] saved {OUT_PNG}")

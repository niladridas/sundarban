"""CMEMS vs HYCOM drifter-validation A/B test.

Same drifter, same release point, same forcing-only-currents config —
just swap the currents NetCDF. The question this answers:

    Is the missed southward feature ~day 7 a CMEMS-specific issue,
    or a generic 1/12° resolution issue that would also affect HYCOM?

If HYCOM shows the same miss → it's a resolution issue (sub-mesoscale
eddy not resolved by either model). If HYCOM catches it → CMEMS-specific
artifact.
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

DRIFTER_CSV = DRIFTERS_DIR / "gdp_300534064293930_2025-04-20_2025-06-01.csv"
CMEMS_NC = CURRENTS_DIR / "custom_2025-04-20_2025-06-01.nc"
HYCOM_NC = CURRENTS_DIR / "custom_2025-04-20_2025-06-01_hycom.nc"
TRAJ_DIR_LOCAL = TRAJ_DIR
OUT_PNG = Path(__file__).resolve().parent.parent / "data" / "drifter_validation_compare.png"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def run_one(currents_nc: Path, label: str, n_days: int, seed_lon: float, seed_lat: float):
    out = TRAJ_DIR_LOCAL / f"drifter_300534064293930_{label}.zarr"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    cfg = SimConfig(
        n_particles=1, runtime_days=n_days,
        dt_minutes=30, output_minutes=60,
        include_stokes=False, include_tides=False,
        include_winds=False, include_settling=False,
    )
    run_simulation(
        currents_nc, cfg, out_path=out, seed=42,
        seed_lons=np.array([seed_lon]),
        seed_lats=np.array([seed_lat]),
    )
    sim_df = traj_to_dataframe(load_trajectories(out)).sort_values("time").reset_index(drop=True)
    return sim_df


def metrics(drf: pd.DataFrame, sim_df: pd.DataFrame) -> dict:
    paired = []
    for _, r in drf.iterrows():
        idx = (sim_df["time"] - r["time"]).abs().idxmin()
        paired.append({
            "time": r["time"],
            "drifter_lat": r["lat"], "drifter_lon": r["lon"],
            "sim_lat": sim_df.loc[idx, "lat"], "sim_lon": sim_df.loc[idx, "lon"],
        })
    pf = pd.DataFrame(paired)
    pf["sep_km"] = haversine_km(pf.drifter_lat, pf.drifter_lon, pf.sim_lat, pf.sim_lon)
    pf["t_days"] = (pf["time"] - pf["time"].iloc[0]).dt.total_seconds() / 86400.0
    drf_lat, drf_lon = pf.drifter_lat.values, pf.drifter_lon.values
    seg = haversine_km(drf_lat[:-1], drf_lon[:-1], drf_lat[1:], drf_lon[1:])
    cum = np.concatenate([[0], np.cumsum(seg)])
    mean_sep = float(pf.sep_km.mean())
    lw = 1.0 - mean_sep / float(np.mean(cum[1:]))
    return {
        "pf": pf,
        "median_sep_km": float(pf.sep_km.median()),
        "p90_sep_km": float(pf.sep_km.quantile(0.9)),
        "max_sep_km": float(pf.sep_km.max()),
        "lw_skill": lw,
        "drifter_path_km": float(cum[-1]),
        "sim_df": sim_df,
    }


# ── Load drifter ─────────────────────────────────────────────────────────────
drf = pd.read_csv(DRIFTER_CSV, skiprows=[1])
drf["time"] = pd.to_datetime(drf["time"]).dt.tz_localize(None)
drf = drf.sort_values("time").reset_index(drop=True)
drf = drf.rename(columns={"latitude": "lat", "longitude": "lon"})
n_days = int(np.ceil((drf.time.max() - drf.time.min()).total_seconds() / 86400))
seed_lon = float(drf.lon.iloc[0])
seed_lat = float(drf.lat.iloc[0])

# ── Run both ─────────────────────────────────────────────────────────────────
console.print("[bold cyan]── CMEMS run ──[/bold cyan]")
sim_cmems = run_one(CMEMS_NC, "cmems", n_days, seed_lon, seed_lat)
console.print("[bold cyan]── HYCOM run ──[/bold cyan]")
sim_hycom = run_one(HYCOM_NC, "hycom", n_days, seed_lon, seed_lat)

m_cmems = metrics(drf, sim_cmems)
m_hycom = metrics(drf, sim_hycom)

console.print()
console.print(f"[bold]Skill comparison ({n_days} days)[/bold]")
hdr = f"  {'metric':<22s} {'CMEMS 1/12°':>14s}   {'HYCOM 1/25°lat':>14s}"
console.print(hdr)
console.print(f"  {'-'*22} {'-'*14}   {'-'*14}")
for label, key, fmt in [
    ("Liu-Weisberg skill", "lw_skill", ".3f"),
    ("median sep (km)", "median_sep_km", ".1f"),
    ("P90 sep (km)", "p90_sep_km", ".1f"),
    ("max sep (km)", "max_sep_km", ".1f"),
]:
    a = format(m_cmems[key], fmt)
    b = format(m_hycom[key], fmt)
    console.print(f"  {label:<22s} {a:>14s}   {b:>14s}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

ax = axes[0]
ax.plot(drf.lon, drf.lat, "-", lw=1.7, color="black", label="GDP drifter (real)")
ax.scatter(drf.lon.iloc[0], drf.lat.iloc[0], marker="o", s=80,
           facecolor="lime", edgecolor="black", zorder=6, label="release")
ax.scatter(drf.lon.iloc[-1], drf.lat.iloc[-1], marker="X", s=80,
           facecolor="black", edgecolor="white", zorder=6, label=f"drifter end (t={n_days}d)")
ax.plot(sim_cmems.lon, sim_cmems.lat, "-", lw=1.4, color="crimson",
        label=f"CMEMS 1/12°  (LW={m_cmems['lw_skill']:.2f})")
ax.scatter(sim_cmems.lon.iloc[-1], sim_cmems.lat.iloc[-1], marker="X", s=70,
           facecolor="crimson", edgecolor="white", zorder=5)
ax.plot(sim_hycom.lon, sim_hycom.lat, "-", lw=1.4, color="navy",
        label=f"HYCOM 1/25° lat (LW={m_hycom['lw_skill']:.2f})")
ax.scatter(sim_hycom.lon.iloc[-1], sim_hycom.lat.iloc[-1], marker="X", s=70,
           facecolor="navy", edgecolor="white", zorder=5)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"GDP drifter {drf.ID.iloc[0]} — CMEMS vs HYCOM")
ax.legend(loc="best", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(m_cmems["pf"].t_days, m_cmems["pf"].sep_km, "-", color="crimson",
        lw=1.4, label=f"CMEMS  (median {m_cmems['median_sep_km']:.0f} km)")
ax.plot(m_hycom["pf"].t_days, m_hycom["pf"].sep_km, "-", color="navy",
        lw=1.4, label=f"HYCOM  (median {m_hycom['median_sep_km']:.0f} km)")
ax.set_xlabel("Days since release")
ax.set_ylabel("Separation from drifter (km)")
ax.set_title("Separation: CMEMS vs HYCOM")
ax.legend(loc="best")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
console.print(f"[green]✓[/green] saved {OUT_PNG}")

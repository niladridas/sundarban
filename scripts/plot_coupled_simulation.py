"""Plot the coupled SPM-seeded simulation.

Three panels:
  1. Initial release pattern (SPM-weighted) overlaid on the SPM map.
  2. Trajectories — every particle's path, colored by time.
  3. Final positions vs initial — dispersion heatmap.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from driftscope.config import SPM_DIR, TRAJ_DIR
from driftscope.viz import load_trajectories, traj_to_dataframe

ROOT = Path(__file__).resolve().parent.parent
SPM_NC = SPM_DIR / "sundarbans_spm_2024-03-08.nc"
TRAJ = TRAJ_DIR / "coupled_spm_seeded_10d.zarr"
OUT_PNG = ROOT / "data" / "coupled_simulation.png"

spm = xr.open_dataset(SPM_NC).load().spm_mg_l
ds = load_trajectories(TRAJ)
df = traj_to_dataframe(ds)
print(f"loaded {len(df):,} particle-observations across {df['traj'].nunique()} trajectories")

# Initial / final per trajectory
first = df.sort_values("time").groupby("traj").head(1)
last = df.sort_values("time").groupby("traj").tail(1)

# Wider bbox covering both release and final positions
lon_min = min(first.lon.min(), last.lon.min()) - 0.1
lon_max = max(first.lon.max(), last.lon.max()) + 0.1
lat_min = min(first.lat.min(), last.lat.min()) - 0.1
lat_max = max(first.lat.max(), last.lat.max()) + 0.1

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

# ── Panel 1: SPM-weighted release pattern over the SPM map
ax = axes[0]
spm_grid = spm.values
im = ax.pcolormesh(
    spm.longitude, spm.latitude, spm_grid,
    norm=LogNorm(vmin=5, vmax=500), cmap="viridis", shading="auto", alpha=0.6,
)
plt.colorbar(im, ax=ax, label="SPM (mg/L)", shrink=0.7)
ax.scatter(first.lon, first.lat, s=3, c="red", alpha=0.6, label=f"{len(first)} releases")
ax.set_xlim(spm.longitude.min(), spm.longitude.max())
ax.set_ylim(spm.latitude.min(), spm.latitude.max())
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Initial release\n(SPM-weighted over Landsat 2024-03-08 retrieval)")
ax.legend(loc="upper right")

# ── Panel 2: Trajectories colored by time
ax = axes[1]
# Convert time to days-since-start for coloring
t0 = df["time"].min()
df["day"] = (df["time"] - t0).dt.total_seconds() / 86400.0
# Plot as scatter (one color per timestep) — fast for 2000 trajs
sc = ax.scatter(df.lon, df.lat, c=df.day, s=0.3, cmap="plasma", alpha=0.4)
plt.colorbar(sc, ax=ax, label="days since release", shrink=0.7)
ax.scatter(first.lon, first.lat, s=8, c="white", edgecolors="black", linewidths=0.3,
           label="release", zorder=5)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Trajectories\n(currents + tides + Stokes + windage + settling)")
ax.legend(loc="upper right")

# ── Panel 3: Final-position density heatmap
ax = axes[2]
n_bins = 80
H, xedges, yedges = np.histogram2d(
    last.lon, last.lat,
    bins=n_bins,
    range=[[lon_min, lon_max], [lat_min, lat_max]],
)
H = H.T  # rows = lat
im = ax.pcolormesh(xedges, yedges, np.where(H > 0, H, np.nan),
                   cmap="hot_r", shading="auto")
plt.colorbar(im, ax=ax, label="particles per cell", shrink=0.7)
ax.scatter(first.lon, first.lat, s=2, c="dodgerblue", alpha=0.4, label="release")
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"Final positions after 10 days\n({len(last)} particles, {n_bins}×{n_bins} grid)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"saved {OUT_PNG}")

# Print dispersion stats
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

merged = first[["traj", "lon", "lat"]].merge(
    last[["traj", "lon", "lat"]], on="traj", suffixes=("_0", "_f"),
)
d_km = haversine_km(merged.lon_0, merged.lat_0, merged.lon_f, merged.lat_f)
print(f"\nDisplacement after 10 days (km):")
print(f"  median: {np.median(d_km):.1f}")
print(f"  P10:    {np.percentile(d_km, 10):.1f}")
print(f"  P90:    {np.percentile(d_km, 90):.1f}")
print(f"  max:    {d_km.max():.1f}")
print(f"  retained: {len(merged)}/{len(first)} particles "
      f"({100*len(merged)/len(first):.1f}% — rest hit OOB and were deleted)")

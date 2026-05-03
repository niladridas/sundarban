"""Visualization utilities.

Three flavors:
- `traj_to_dataframe` — flatten Zarr trajectories to a long-format pandas DF
  suitable for pydeck/plotly
- `accumulation_grid` — bin final particle positions into a 2D heatmap
- `plot_static` — quick matplotlib snapshot for notebooks
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_trajectories(traj_path: Path) -> xr.Dataset:
    """Open a Parcels trajectory output (Zarr or NetCDF).

    Parcels v3 writes asymmetric chunks when DeleteOOB removes particles
    mid-run: `z` (depth) ends up one obs step shorter than lat/lon/time and
    xarray refuses to open the Zarr. We don't need depth for 2D viz, so we
    fall back to dropping `z`.
    """
    p = Path(traj_path)
    if p.suffix == ".zarr" or p.is_dir():
        try:
            return xr.open_zarr(p)
        except ValueError as e:
            if "obs" in str(e) and "z" in str(e):
                return xr.open_zarr(p, drop_variables=["z"])
            raise
    return xr.open_dataset(p)


def traj_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Long-format DF with columns: traj, obs, time, lon, lat.

    Drops NaN rows (deleted/out-of-bounds particles).
    """
    df = ds[["lon", "lat", "time"]].to_dataframe().reset_index()
    # Parcels v3 names the particle dimension "trajectory"; older code/docs use "traj".
    df = df.rename(columns={"trajectory": "traj"})
    df = df.dropna(subset=["lon", "lat"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def accumulation_grid(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    n_bins: int = 80,
    only_final: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of particle positions.

    Parameters
    ----------
    df : long-format trajectories
    bbox : (lon_min, lat_min, lon_max, lat_max)
    n_bins : grid resolution
    only_final : if True, only count each particle's last position

    Returns
    -------
    H : (n_bins, n_bins) counts
    xedges, yedges : bin edges
    """
    if only_final:
        # last valid position per trajectory
        df = df.sort_values("time").groupby("traj").tail(1)

    lon_min, lat_min, lon_max, lat_max = bbox
    H, xedges, yedges = np.histogram2d(
        df["lon"].values,
        df["lat"].values,
        bins=n_bins,
        range=[[lon_min, lon_max], [lat_min, lat_max]],
    )
    return H.T, xedges, yedges  # transpose so rows=lat


def plot_static(traj_path: Path, out_png: Path | None = None) -> None:
    """Quick matplotlib snapshot of trajectories for notebooks/reports."""
    import matplotlib.pyplot as plt

    ds = load_trajectories(traj_path)
    df = traj_to_dataframe(ds)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # Trajectories
    for tid, g in df.groupby("traj"):
        ax.plot(g["lon"], g["lat"], lw=0.4, alpha=0.4, color="#3b82f6")

    # Start (green) & end (red)
    starts = df.sort_values("time").groupby("traj").head(1)
    ends = df.sort_values("time").groupby("traj").tail(1)
    ax.scatter(starts["lon"], starts["lat"], s=4, c="#10b981", label="start", zorder=3)
    ax.scatter(ends["lon"], ends["lat"], s=4, c="#ef4444", label="end", zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Lagrangian trajectories — {len(df['traj'].unique())} particles")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"saved {out_png}")
    else:
        plt.show()

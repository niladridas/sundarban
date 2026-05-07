"""Particle-trajectory animation for the GBM/Sundarbans system.

Renders Lagrangian particle positions over time as MP4 (or GIF), with a
Cartopy basemap (coastlines + land mask). Designed to make sediment-plume
dynamics visually compelling — different rivers in different colors, an
optional fading "tail" for each particle, and the data datetime overlaid.

Usage from a script:

    from driftscope.animate import animate_trajectories
    animate_trajectories(
        traj_zarr=Path("data/trajectories/foo.zarr"),
        out_path=Path("data/animations/foo.mp4"),
        bbox=(87.5, 19.0, 92.5, 23.0),
        river_idx=river_idx,            # optional, for per-river coloring
        river_names=["Hooghly", ...],   # legend
        fps=15, tail_steps=8,
    )
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rich.console import Console

from .viz import load_trajectories

console = Console()

# Distinct, perceptually-spaced colors for ~12 rivers. Comes from the
# matplotlib tab20 + a few hand-picked highlights, ordered W → E so
# adjacent rivers along the coast get adjacent hues.
RIVER_COLORS = [
    "#e41a1c",  # red       — Hooghly
    "#ff7f00",  # orange    — Saptamukhi
    "#ffd92f",  # yellow    — Thakuran
    "#a6d854",  # lime      — Matla
    "#4daf4a",  # green     — Raimangal
    "#1b9e77",  # teal      — Harinbhanga
    "#377eb8",  # blue      — Pussur-Sibsa
    "#7570b3",  # purple    — Baleswar
    "#984ea3",  # violet    — Tetulia
    "#e7298a",  # magenta   — Lower-Meghna
    "#a65628",  # brown     — Meghna-Main
    "#f781bf",  # pink      — Sandwip-Channel
]


def animate_trajectories(
    traj_zarr: Path,
    out_path: Path,
    bbox: tuple[float, float, float, float],
    river_idx: np.ndarray | None = None,
    river_names: Sequence[str] | None = None,
    fps: int = 12,
    tail_steps: int = 6,
    point_size: float = 8.0,
    title_prefix: str = "GBM sediment dispersion",
    bathy_nc: Path | None = None,
) -> Path:
    """Animate a Parcels trajectory zarr to MP4 with a Cartopy basemap.

    Parameters
    ----------
    traj_zarr
        Parcels output (Zarr) — must have lon, lat, time per particle.
    out_path
        Output video path. `.mp4` uses ffmpeg; `.gif` uses Pillow.
    bbox
        (lon_min, lat_min, lon_max, lat_max) — the visible extent.
    river_idx
        Optional (n_particles,) array assigning each particle to a river
        (used for per-river coloring). If None, all particles are blue.
    river_names
        Optional list of river names for the legend. Length must match
        max(river_idx)+1 if provided.
    fps
        Frames per second in the output video.
    tail_steps
        How many previous output steps to draw as a fading tail. 0 = no tail.
    point_size
        Marker size in matplotlib points² for current positions.
    bathy_nc
        Optional NetCDF with a depth grid (e.g. CMEMS bathymetry). If given,
        adds shaded depth contours to the basemap.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import imageio.v3 as iio

    ds = load_trajectories(traj_zarr)
    # Parcels v3 axes: trajectory (n_part), obs (n_step). Positions are NaN
    # for steps before/after a particle exists; we just plot non-NaN.
    lon = ds.lon.values   # (n_part, n_obs)
    lat = ds.lat.values
    # Time is stored per-particle (also (n_part, n_obs)); for the per-step
    # title timestamp, find the first non-NaT particle's time at each step
    # (if particle 0 was deleted, its later times are NaT).
    time_arr = ds.time.values
    if time_arr.ndim == 2:
        t_list = []
        for s in range(time_arr.shape[1]):
            valid = pd.notna(time_arr[:, s])
            if valid.any():
                t_list.append(pd.Timestamp(time_arr[valid, s][0]))
            else:
                # No particle has a time at this step — interpolate from nearest
                t_list.append(pd.NaT)
        t = pd.to_datetime(t_list)
    else:
        t = pd.to_datetime(time_arr)
    n_part, n_obs = lon.shape
    console.print(f"[cyan]·[/cyan] {n_part} particles × {n_obs} time steps")

    # Build per-particle colors
    if river_idx is not None:
        ri = np.asarray(river_idx, dtype=int)
        cmap = mcolors.ListedColormap(RIVER_COLORS[: ri.max() + 1])
        colors = cmap(ri / max(1, ri.max()))
    else:
        colors = np.tile(np.array([[0.1, 0.4, 0.9, 1.0]]), (n_part, 1))
        cmap = None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up the figure with a Cartopy axes
    fig = plt.figure(figsize=(11, 9), dpi=110)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Cartopy wants [lon_min, lon_max, lat_min, lat_max] (not bbox order!)
    lon_min, lat_min, lon_max, lat_max = bbox
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Use 50m-resolution Natural Earth features — 110m is too coarse for the
    # Sundarbans coast where the delta channels are <10 km wide.
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0c1e2e", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#1d3a1d",
                   edgecolor="#0a1f0a", linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#5d7a5d",
                   linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="#2a5a8a",
                   linewidth=0.5, zorder=2)
    gl = ax.gridlines(draw_labels=True, alpha=0.25, linewidth=0.3, color="white")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#888"}
    gl.ylabel_style = {"size": 8, "color": "#888"}

    # Optional bathymetry contours
    if bathy_nc is not None and bathy_nc.exists():
        bds = xr.open_dataset(bathy_nc)
        if "depth" in bds:
            depth = bds["depth"].squeeze()
            ax.contour(depth.longitude, depth.latitude, depth,
                       levels=[20, 50, 200, 1000, 2000], colors="#3a5b8a",
                       linewidths=0.5, alpha=0.5,
                       transform=ccrs.PlateCarree(), zorder=2)

    # River legend, if names given
    if river_names is not None:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=RIVER_COLORS[i], label=name)
                   for i, name in enumerate(river_names)]
        leg = ax.legend(handles=handles, loc="lower left", fontsize=7,
                        facecolor="#0c1e2e", edgecolor="#888",
                        labelcolor="white", title="Source",
                        title_fontproperties={"size": 8, "weight": "bold"},
                        framealpha=0.85)
        leg.get_title().set_color("white")

    # Title & timestamp text
    title_obj = ax.set_title("", color="white", fontsize=11, pad=10)
    fig.patch.set_facecolor("#06121b")
    ax.set_facecolor("#06121b")

    # Pre-create scatter artists; updated each frame
    scat = ax.scatter([], [], s=point_size, c="white",
                      edgecolors="none", transform=ccrs.PlateCarree(),
                      zorder=5)
    tail_scats = []
    for k in range(tail_steps):
        alpha = 0.18 * (1 - k / max(1, tail_steps))
        s = ax.scatter([], [], s=point_size * 0.7, c="white",
                       edgecolors="none", alpha=alpha,
                       transform=ccrs.PlateCarree(), zorder=4)
        tail_scats.append(s)

    # Render frames to memory and write the video
    console.print(f"[cyan]→[/cyan] rendering {n_obs} frames @ {fps} fps")
    frames = []
    for step in range(n_obs):
        # Current positions
        valid = np.isfinite(lon[:, step]) & np.isfinite(lat[:, step])
        pts = np.column_stack([lon[valid, step], lat[valid, step]])
        scat.set_offsets(pts if len(pts) else np.empty((0, 2)))
        scat.set_color(colors[valid] if len(pts) else "white")

        # Tail
        for k, s in enumerate(tail_scats, start=1):
            ts = step - k
            if ts < 0:
                s.set_offsets(np.empty((0, 2)))
                continue
            v = np.isfinite(lon[:, ts]) & np.isfinite(lat[:, ts])
            tp = np.column_stack([lon[v, ts], lat[v, ts]])
            s.set_offsets(tp if len(tp) else np.empty((0, 2)))
            s.set_color(colors[v] if len(tp) else "white")

        # Title with timestamp (gracefully handle steps with all-NaT times)
        if pd.notna(t[step]):
            ts_str = t[step].strftime("%Y-%m-%d %H:%M UTC")
            days_since = (t[step] - t[0]).total_seconds() / 86400.0
            title_obj.set_text(f"{title_prefix}   ·   {ts_str}   ·   t = {days_since:.1f} d")
        else:
            title_obj.set_text(f"{title_prefix}   ·   step {step}/{n_obs}")

        fig.canvas.draw()
        # Use buffer_rgba and drop the alpha channel — works on Agg backend
        # for Matplotlib 3.8+ where tostring_rgb was removed.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())

        if (step + 1) % 20 == 0 or step == n_obs - 1:
            console.print(f"   {step+1}/{n_obs}")

    plt.close(fig)

    # Write video
    if out_path.suffix == ".mp4":
        iio.imwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
    else:
        # GIF fallback
        iio.imwrite(out_path, frames, duration=1000 / fps, loop=0)
    console.print(f"[green]✓[/green] {out_path} ({len(frames)} frames, {out_path.stat().st_size/1e6:.1f} MB)")
    return out_path

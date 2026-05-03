"""Eulerian diagnostics of the surface velocity field.

Computes the same quantities shown in the GOFLOW (Klein et al.) figure:
speed, log|∇V|, vorticity (ζ), divergence (δ), all on the native lat/lon
grid. ζ/|f| and δ/|f| are dimensionless Rossby-number-like ratios — values
above ~0.5 mark places where ageostrophic dynamics dominate (fronts, eddy
edges, intense filaments).

The math is straightforward: gradient → curl/divergence → normalize by
Coriolis. The only subtlety is converting `np.gradient` outputs from
per-degree to per-meter using local cos(latitude) for longitude spacing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

R_EARTH = 6_371_000.0    # m
OMEGA = 7.2921e-5        # rad/s


def compute_diagnostics(
    u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> dict[str, np.ndarray]:
    """All fields returned on the same (lat, lon) grid as `u`, `v`.

    Parameters
    ----------
    u, v
        2D velocity components shape (n_lat, n_lon), m/s.
    lats, lons
        1D coordinate arrays in degrees.

    Returns
    -------
    dict with keys: speed, vorticity, divergence, grad_mag,
                    rossby_vort, rossby_div, f
    """
    LAT2D = np.broadcast_to(lats[:, None], u.shape)
    cos_lat = np.cos(np.deg2rad(LAT2D))

    deg2m_lat = R_EARTH * np.pi / 180.0
    deg2m_lon = deg2m_lat * cos_lat

    du_dlat, du_dlon = np.gradient(u, lats, lons)
    dv_dlat, dv_dlon = np.gradient(v, lats, lons)

    du_dx = du_dlon / deg2m_lon
    du_dy = du_dlat / deg2m_lat
    dv_dx = dv_dlon / deg2m_lon
    dv_dy = dv_dlat / deg2m_lat

    speed = np.sqrt(u ** 2 + v ** 2)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    grad_mag = np.sqrt(du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2)

    f = 2 * OMEGA * np.sin(np.deg2rad(LAT2D))
    # Avoid divide-by-zero across the equator
    f_safe = np.where(np.abs(f) < 1e-10, np.nan, f)

    return {
        "speed": speed,
        "vorticity": vorticity,
        "divergence": divergence,
        "grad_mag": grad_mag,
        "rossby_vort": vorticity / np.abs(f_safe),
        "rossby_div": divergence / np.abs(f_safe),
        "f": f,
    }


def _style_panel(ax, title: str, fg: str) -> None:
    ax.set_title(title, color=fg, fontsize=11, weight="600")
    ax.set_xlabel("Longitude (°)", color=fg, fontsize=9)
    ax.set_ylabel("Latitude (°)", color=fg, fontsize=9)
    ax.tick_params(colors=fg, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg)


def plot_diagnostics(
    u: np.ndarray,
    v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    time_str: str | None = None,
    dark: bool = False,
) -> "plt.Figure":
    """Render a 2×2 panel: speed+vectors, log|∇V|, ζ/f, δ/f."""
    d = compute_diagnostics(u, v, lats, lons)

    bg = "#0f172a" if dark else "#ffffff"
    fg = "#f8fafc" if dark else "#0f172a"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=bg)
    for ax in axes.flat:
        ax.set_facecolor(bg)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # (a) Speed + velocity vectors
    ax = axes[0, 0]
    speed_max = float(np.nanpercentile(d["speed"], 99)) if np.isfinite(d["speed"]).any() else 1.0
    im = ax.imshow(
        d["speed"], extent=extent, origin="lower",
        cmap="viridis", aspect="auto", vmin=0, vmax=max(speed_max, 1e-3),
    )
    step = max(1, len(lons) // 25)
    LON, LAT = np.meshgrid(lons, lats)
    ax.quiver(
        LON[::step, ::step], LAT[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color="white", scale=18, width=0.0022, alpha=0.6,
    )
    cbar = plt.colorbar(im, ax=ax, label="|V|  (m s⁻¹)")
    cbar.ax.yaxis.label.set_color(fg)
    cbar.ax.tick_params(colors=fg)
    _style_panel(ax, "(a) Speed + velocity vectors", fg)

    # (b) log|∇V|
    ax = axes[0, 1]
    log_grad = np.log10(d["grad_mag"] + 1e-12)
    finite = log_grad[np.isfinite(log_grad)]
    if finite.size:
        vmin, vmax = float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 99))
    else:
        vmin, vmax = -8, -4
    im = ax.imshow(
        log_grad, extent=extent, origin="lower",
        cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax, label="log₁₀ |∇V|")
    cbar.ax.yaxis.label.set_color(fg)
    cbar.ax.tick_params(colors=fg)
    _style_panel(ax, "(b) log|∇V|  —  fronts & gradients", fg)

    # (c) ζ/|f|
    ax = axes[1, 0]
    im = ax.imshow(
        d["rossby_vort"], extent=extent, origin="lower",
        cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1,
    )
    cbar = plt.colorbar(im, ax=ax, label="ζ / |f|")
    cbar.ax.yaxis.label.set_color(fg)
    cbar.ax.tick_params(colors=fg)
    _style_panel(ax, "(c) Relative vorticity / Coriolis", fg)

    # (d) δ/|f|
    ax = axes[1, 1]
    im = ax.imshow(
        d["rossby_div"], extent=extent, origin="lower",
        cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2,
    )
    cbar = plt.colorbar(im, ax=ax, label="δ / |f|")
    cbar.ax.yaxis.label.set_color(fg)
    cbar.ax.tick_params(colors=fg)
    _style_panel(ax, "(d) Divergence / Coriolis", fg)

    if time_str:
        fig.suptitle(
            f"Velocity field diagnostics  ·  {time_str}",
            fontsize=13, weight="bold", color=fg,
        )
    plt.tight_layout()
    return fig

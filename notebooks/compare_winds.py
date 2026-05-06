"""Compare currents-only vs currents+windage trajectories.

Same pattern as compare_tides.py: paired-particle displacement metric
plus a 3-panel figure (no-wind cloud, with-wind cloud, displacement
histogram). The expected windage signature is direction-coherent
deflection (whichever way the wind blows), not the symmetric oscillation
that tides produce.

Usage:
    python notebooks/compare_winds.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "data" / "trajectories"

NO_WIND = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01_traj.zarr"
WIND = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01__wind3.0pct_traj.zarr"
WINDAGE_PCT = 3.0


def load(p: Path) -> xr.Dataset:
    return xr.open_zarr(p)


def pair_displacement(a: xr.Dataset, b: xr.Dataset) -> np.ndarray:
    n = min(a.sizes["trajectory"], b.sizes["trajectory"])
    nt = min(a.sizes["obs"], b.sizes["obs"])
    la = np.deg2rad(a.lat.values[:n, :nt])
    lo = np.deg2rad(a.lon.values[:n, :nt])
    lb = np.deg2rad(b.lat.values[:n, :nt])
    ob = np.deg2rad(b.lon.values[:n, :nt])
    dlat = lb - la
    dlon = ob - lo
    h = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    no_w = load(NO_WIND)
    with_w = load(WIND)
    print(f"no-wind:   {no_w.sizes['trajectory']} particles, {no_w.sizes['obs']} obs")
    print(f"with-wind: {with_w.sizes['trajectory']} particles, "
          f"{with_w.sizes['obs']} obs ({WINDAGE_PCT}% windage)")

    disp = pair_displacement(no_w, with_w)
    finite = np.isfinite(disp)
    last_idx = np.where(finite.any(axis=1),
                        finite.shape[1] - 1 - finite[:, ::-1].argmax(axis=1),
                        -1)
    rows = np.where(last_idx >= 0)[0]
    end_disp = disp[rows, last_idx[rows]]
    end_disp = end_disp[np.isfinite(end_disp)]
    all_disp = disp[finite]

    print(f"\nPaired displacement (km) — currents-only vs currents+{WINDAGE_PCT}% wind:")
    print(f"  particles with valid pair: {len(end_disp)}/{disp.shape[0]}")
    print(f"  last-valid median:  {np.median(end_disp):.3f} km")
    print(f"  last-valid p95:     {np.percentile(end_disp, 95):.3f} km")
    print(f"  last-valid max:     {end_disp.max():.3f} km")
    print(f"  full-trajectory median over all (i,t): {np.median(all_disp):.3f} km")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_show = min(150, no_w.sizes["trajectory"])
    for ax, ds, title in (
        (axes[0], no_w,  "Currents only"),
        (axes[1], with_w, f"Currents + {WINDAGE_PCT}% windage"),
    ):
        for i in range(n_show):
            x = ds.lon.values[i]
            y = ds.lat.values[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 1:
                ax.plot(x[m], y[m], "-", lw=0.4, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)

    axes[2].hist(end_disp, bins=40, color="#a855f7", alpha=0.85)
    axes[2].axvline(np.median(end_disp), color="r", ls="--",
                    label=f"median {np.median(end_disp):.1f} km")
    axes[2].set_xlabel("Per-particle displacement (km)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Wind-induced displacement at end of sim ({WINDAGE_PCT}%)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Windage validation: same release, same RNG, "
        f"with vs without ERA5 10m wind × {WINDAGE_PCT}%",
        fontsize=13,
    )
    fig.tight_layout()
    out = ROOT / "data" / "winds_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

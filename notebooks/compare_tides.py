"""Compare currents-only vs currents+tides trajectories on the same release.

Both sims use seed=42, so trajectory[i] from run A and run B started at the
same point. Pairing them lets us compute the per-particle displacement that
the tide forcing introduces, separate from the currents-only background.

Usage:
    python notebooks/compare_tides.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "data" / "trajectories"

NO_TIDES = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01_traj.zarr"
TIDES = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01__tides_traj.zarr"


def load(p: Path) -> xr.Dataset:
    return xr.open_zarr(p)


def pair_displacement(a: xr.Dataset, b: xr.Dataset) -> np.ndarray:
    """Great-circle distance (km) between paired particles at each time."""
    n = min(a.sizes["trajectory"], b.sizes["trajectory"])
    nt = min(a.sizes["obs"], b.sizes["obs"])
    la = np.deg2rad(a.lat.values[:n, :nt])
    lo = np.deg2rad(a.lon.values[:n, :nt])
    lb = np.deg2rad(b.lat.values[:n, :nt])
    ob = np.deg2rad(b.lon.values[:n, :nt])
    # Haversine
    dlat = lb - la
    dlon = ob - lo
    h = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    no_t = load(NO_TIDES)
    with_t = load(TIDES)
    print(f"no-tides:  {no_t.sizes['trajectory']} particles, {no_t.sizes['obs']} obs")
    print(f"with-tides:{with_t.sizes['trajectory']} particles, {with_t.sizes['obs']} obs")

    disp = pair_displacement(no_t, with_t)  # (n_particles, nt)
    finite = np.isfinite(disp)

    # Last-valid-position displacement per particle: take the latest timestep
    # where BOTH runs still hold a finite position for that particle.
    last_idx = np.where(finite.any(axis=1),
                        finite.shape[1] - 1 - finite[:, ::-1].argmax(axis=1),
                        -1)
    rows = np.where(last_idx >= 0)[0]
    end_disp = disp[rows, last_idx[rows]]
    end_disp = end_disp[np.isfinite(end_disp)]
    all_disp = disp[finite]

    print(f"\nPaired displacement (km) — currents-only vs currents+tides:")
    print(f"  particles with valid pair: {len(end_disp)}/{disp.shape[0]}")
    print(f"  last-valid median:  {np.median(end_disp):.3f} km")
    print(f"  last-valid p95:     {np.percentile(end_disp, 95):.3f} km")
    print(f"  last-valid max:     {end_disp.max():.3f} km")
    print(f"  full-trajectory median over all (i,t): {np.median(all_disp):.3f} km")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Trajectory clouds
    n_show = min(150, no_t.sizes["trajectory"])
    for ax, ds, title in (
        (axes[0], no_t, "Currents only"),
        (axes[1], with_t, "Currents + tides"),
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

    # Displacement histogram
    axes[2].hist(end_disp, bins=40, color="#3b82f6", alpha=0.85)
    axes[2].axvline(np.median(end_disp), color="r", ls="--",
                    label=f"median {np.median(end_disp):.1f} km")
    axes[2].set_xlabel("Per-particle displacement (km)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Tide-induced displacement at end of sim")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "Tides validation: same release, same RNG, with vs without M2/S2/K1/O1...",
        fontsize=13,
    )
    fig.tight_layout()
    out = ROOT / "data" / "tides_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

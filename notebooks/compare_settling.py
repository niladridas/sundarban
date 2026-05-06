"""Compare currents-only vs currents+settling trajectories.

Different from compare_tides/compare_winds: settling doesn't deflect
particles horizontally — it sinks them. So the comparison metric is
particle DEPTH over time, not paired horizontal displacement.

Validation expectation for 10 µm silt: w_s ≈ 7.6 m/day Stokes settling,
so after 10 days particles should reach ~76 m depth in the absence of
vertical mixing or resuspension (which we don't model).

Usage:
    python notebooks/compare_settling.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from driftscope...` works when run as a script
TRAJ_DIR = ROOT / "data" / "trajectories"

NO_SETTLE = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01_traj.zarr"
SETTLE = TRAJ_DIR / "sundarbans_delta_2026-04-17_2026-05-01__settle10um_traj.zarr"
DIAMETER_UM = 10.0


def load(p: Path) -> xr.Dataset:
    return xr.open_zarr(p)


def main() -> None:
    no_s = load(NO_SETTLE)
    with_s = load(SETTLE)
    print(f"no-settling:   {no_s.sizes['trajectory']} particles, "
          f"{no_s.sizes['obs']} obs")
    print(f"with-settling: {with_s.sizes['trajectory']} particles, "
          f"{with_s.sizes['obs']} obs ({DIAMETER_UM} µm Stokes)")

    z_off = no_s.z.values     # (n_particles, n_obs) — depths in meters, +down
    z_on  = with_s.z.values
    t_on  = with_s.time.values
    # Convert times to elapsed days for plotting
    t0 = np.nanmin(t_on)
    elapsed_days = (t_on - t0) / np.timedelta64(1, "D")
    elapsed_days = np.nanmean(elapsed_days, axis=0)  # ~uniform across particles

    print(f"\nDepth stats (m, +down) — currents+settling run:")
    finite_z_on = z_on[np.isfinite(z_on)]
    print(f"  initial depth: {np.nanmedian(z_on[:, 0]):.2f}")
    print(f"  end depth (last valid per particle):"
          f"  median {np.nanmedian(z_on[:, -1]):.2f},  "
          f"max {np.nanmax(finite_z_on):.2f}")

    # Predicted final depth at constant w_s for comparison
    from driftscope.kernels import stokes_settling_velocity
    w_s = stokes_settling_velocity(DIAMETER_UM)
    runtime_d = float(elapsed_days[-1])
    predicted_final = w_s * 86400 * runtime_d
    print(f"  Stokes prediction: {predicted_final:.2f} m at {runtime_d:.1f} d")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: depth over time — no settling
    n_show = min(150, no_s.sizes["trajectory"])
    for i in range(n_show):
        z = z_off[i]
        m = np.isfinite(z)
        if m.sum() > 1:
            axes[0].plot(elapsed_days[m], z[m], "-", lw=0.5, alpha=0.4,
                         color="#3b82f6")
    axes[0].set_title("Currents only")
    axes[0].set_xlabel("Elapsed time (days)")
    axes[0].set_ylabel("Depth (m, +down)")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3)

    # Panel 2: depth over time — with settling
    for i in range(n_show):
        z = z_on[i]
        m = np.isfinite(z)
        if m.sum() > 1:
            axes[1].plot(elapsed_days[m], z[m], "-", lw=0.5, alpha=0.4,
                         color="#10b981")
    # Overlay the analytic Stokes prediction
    axes[1].plot(elapsed_days, w_s * 86400 * elapsed_days, "k--", lw=1.5,
                 label=f"Stokes: w_s×t  ({w_s*86400:.2f} m/day)")
    axes[1].set_title(f"Currents + settling ({DIAMETER_UM} µm silt)")
    axes[1].set_xlabel("Elapsed time (days)")
    axes[1].set_ylabel("Depth (m, +down)")
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    # Panel 3: distribution of final depths
    final_z_on  = np.array([row[np.isfinite(row)][-1]
                             for row in z_on if np.isfinite(row).any()])
    final_z_off = np.array([row[np.isfinite(row)][-1]
                             for row in z_off if np.isfinite(row).any()])
    axes[2].hist(final_z_off, bins=30, alpha=0.6, label="no settling",
                 color="#3b82f6")
    axes[2].hist(final_z_on, bins=30, alpha=0.6, label=f"{DIAMETER_UM} µm",
                 color="#10b981")
    axes[2].axvline(predicted_final, color="k", ls="--",
                    label=f"Stokes prediction {predicted_final:.0f} m")
    axes[2].set_xlabel("Final depth (m, +down)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Final-depth distribution")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Settling validation: same release, same RNG, "
        f"with vs without {DIAMETER_UM} µm Stokes settling",
        fontsize=13,
    )
    fig.tight_layout()
    out = ROOT / "data" / "settling_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

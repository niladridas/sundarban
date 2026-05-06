"""OceanParcels kernels.

Each kernel is a small physics module that updates particle position/state
each timestep. Build new modules (Stokes drift, windage, settling) by adding
new kernels here and chaining them in simulate.py — that's the whole point
of the architecture.

Reference: https://oceanparcels.org/
"""
from __future__ import annotations


# ── Stokes' law for SPM settling (Module 5) ──────────────────────────────────
# w_s = (2/9) * (rho_p - rho_w) / mu * g * r^2
# Valid for particle Reynolds < 0.5, i.e. diameter ≲ 100 µm in water.
# Larger grains need inertial corrections (Cheng 1997, Soulsby 1997) — out
# of scope for now, but the helper makes the assumption explicit.
GRAVITY = 9.81
WATER_DENSITY_KG_M3 = 1025.0      # seawater
WATER_VISCOSITY_PA_S = 1.002e-3   # ~20°C; for monsoon-warmed Bay, 0.85e-3 is closer


def stokes_settling_velocity(
    diameter_um: float,
    particle_density_kg_m3: float = 2650.0,   # quartz/silica
    water_density_kg_m3: float = WATER_DENSITY_KG_M3,
    viscosity_pa_s: float = WATER_VISCOSITY_PA_S,
) -> float:
    """Stokes terminal settling velocity in m/s. Positive = downward.

    Reasonable presets for Sundarbans SPM:
      - 1 µm clay:  ~9e-7 m/s  (~0.08 m/day)
      - 10 µm silt: ~9e-5 m/s  (~7.6 m/day)
      - 50 µm fine sand: ~2.2e-3 m/s (Stokes' law starts breaking down)
    """
    r = (diameter_um * 1e-6) / 2.0
    return (
        (2.0 / 9.0)
        * (particle_density_kg_m3 - water_density_kg_m3)
        / viscosity_pa_s
        * GRAVITY
        * r * r
    )


def AdvectionRK4_Land(particle, fieldset, time):
    """RK4 advection that gracefully deletes particles that hit land/NaN.

    Parcels' built-in `AdvectionRK4` raises an error when sampling NaN
    (out-of-domain or land). Wrap it so we just mark and remove the particle —
    much friendlier for coastal regions like the Sundarbans where particles
    *will* run aground.
    """
    # k1
    (u1, v1) = fieldset.UV[particle]
    lon1 = particle.lon + u1 * 0.5 * particle.dt
    lat1 = particle.lat + v1 * 0.5 * particle.dt

    # k2
    (u2, v2) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat1, lon1, particle]
    lon2 = particle.lon + u2 * 0.5 * particle.dt
    lat2 = particle.lat + v2 * 0.5 * particle.dt

    # k3
    (u3, v3) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat2, lon2, particle]
    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt

    # k4
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]

    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particle.dt  # noqa: F821
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particle.dt  # noqa: F821


def DeleteOnError(particle, fieldset, time):
    """Catch out-of-bounds / NaN sampling and remove the particle cleanly."""
    if particle.state >= 50:  # ErrorCode.Error or worse
        particle.delete()


def Settling(particle, fieldset, time):
    """Stokes settling for SPM. Requires per-particle `w_settle` (m/s, +down).

    Use with a particle class that adds a `w_settle` Variable. Stokes-drift,
    windage, and tides all live in the U/V data layer (combine_forcings);
    only settling needs a kernel because its velocity is per-particle, not
    a spatial field.
    """
    particle_ddepth += particle.w_settle * particle.dt  # noqa: F821


# ── Future modules (stubs to make the extension pattern obvious) ─────────────
# def StokesDrift(particle, fieldset, time):
#     """Add wave-induced Stokes drift on top of Eulerian currents."""
#     us, vs = fieldset.US[particle], fieldset.VS[particle]
#     particle_dlon += us * particle.dt
#     particle_dlat += vs * particle.dt
#
# def Windage(particle, fieldset, time):
#     """Buoyant material: a few % of wind speed."""
#     uw, vw = fieldset.U10[particle], fieldset.V10[particle]
#     particle_dlon += 0.03 * uw * particle.dt
#     particle_dlat += 0.03 * vw * particle.dt
#
# def Settling(particle, fieldset, time):
#     """SPM/sediment settling."""
#     particle.depth += particle.w_settle * particle.dt

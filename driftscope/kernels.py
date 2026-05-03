"""OceanParcels kernels.

Each kernel is a small physics module that updates particle position/state
each timestep. Build new modules (Stokes drift, windage, settling) by adding
new kernels here and chaining them in simulate.py — that's the whole point
of the architecture.

Reference: https://oceanparcels.org/
"""
from __future__ import annotations


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

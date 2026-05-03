"""Configuration: regions of interest, default parameters, paths.

Region-agnostic by design: pass any bbox to the simulate/fetch functions.
The PRESETS dict gives nice starting points for common areas.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CURRENTS_DIR = DATA / "currents"
WAVES_DIR = DATA / "waves"
TRAJ_DIR = DATA / "trajectories"

CURRENTS_DIR.mkdir(parents=True, exist_ok=True)
WAVES_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# ── Region presets ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Region:
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    # Optional offshore release point — for regions whose bbox center is on land
    # (deltas, river mouths). Falls back to bbox center via default_seed.
    seed_lon: float | None = None
    seed_lat: float | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.lon_min + self.lon_max) / 2, (self.lat_min + self.lat_max) / 2)

    @property
    def default_seed(self) -> tuple[float, float]:
        if self.seed_lon is not None and self.seed_lat is not None:
            return (self.seed_lon, self.seed_lat)
        return self.center


PRESETS: dict[str, Region] = {
    # Bbox center is in the mangrove delta (land); seed offshore in open Bay of Bengal.
    "sundarbans": Region(
        "Sundarbans Delta", 88.0, 90.5, 21.0, 22.8,
        seed_lon=89.0, seed_lat=21.2,
    ),
    # Wider view spanning the full upper Bay of Bengal — Hooghly mouth → Chattogram coast.
    # Best paired with seed_mode="bbox" (basin fill) to show drift across the whole basin.
    "northern_bay_of_bengal": Region(
        "Northern Bay of Bengal", 86.5, 92.5, 19.5, 23.0,
        seed_lon=89.5, seed_lat=20.5,
    ),
    "bay_of_bengal": Region("Bay of Bengal (broad)", 80.0, 95.0, 10.0, 23.0),
    # Bbox center sits on the river mouth; seed east-south in open Bay water.
    "hooghly_mouth": Region(
        "Hooghly River Mouth", 87.8, 89.0, 21.2, 22.2,
        seed_lon=88.6, seed_lat=21.4,
    ),
    "gulf_stream": Region("Gulf Stream", -75.0, -60.0, 35.0, 44.0),
    "california": Region("California Current", -127.0, -117.0, 32.0, 42.0),
}

DEFAULT_REGION = "sundarbans"


# ── CMEMS dataset ────────────────────────────────────────────────────────────
# Global Ocean Physics Analysis & Forecast — surface currents, daily, 1/12°
# https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024
CMEMS_DATASET_ID = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
CMEMS_VARIABLES = ["uo", "vo"]  # zonal & meridional velocity (m/s)
CMEMS_DEPTH = 0.494025  # surface layer (m) — exact dataset coord, must be ≥ 0.49402499


# ── ERA5 wave dataset (for Stokes drift) ─────────────────────────────────────
# ECMWF ERA5 hourly single-level reanalysis. Free CDS API key required —
# see .env.example. We derive Stokes drift from H_s, T_m, and mean direction
# using the deep-water surface formula: U_s ≈ (2π)³·H²/(8g·T³).
ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",  # swh, m
    "mean_wave_period",                                     # mwp, s
    "mean_wave_direction",                                  # mwd, deg (from)
]
ERA5_TIMES = [f"{h:02d}:00" for h in range(0, 24, 6)]  # 6-hourly snapshots


# ── Simulation defaults ──────────────────────────────────────────────────────
@dataclass
class SimConfig:
    n_particles: int = 500
    runtime_days: int = 10
    dt_minutes: int = 30        # advection step
    output_minutes: int = 180   # save trajectory every N minutes
    seed_lon: float | None = None  # if None, uses region center
    seed_lat: float | None = None
    seed_radius_deg: float = 0.15  # particles seeded in a disk
    # "disk": release in a small disk around (seed_lon, seed_lat).
    # "bbox": uniformly fill every ocean cell of the data bbox (basin-fill).
    seed_mode: str = "disk"
    include_stokes: bool = False   # add ERA5-derived Stokes drift to advection
    # Multiplier on derived Stokes drift magnitude. Different sources put the
    # Phillips-approximation prefactor anywhere from π³ (textbook narrow-band)
    # to 8π³ (OpenDrift). 1.0 = our baseline; ~8.0 matches the stronger end.
    stokes_scale: float = 1.0

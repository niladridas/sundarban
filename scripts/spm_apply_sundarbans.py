"""Apply the AquaMatch-trained CBR model to a Sundarbans Landsat scene.

Produces the first DriftScope SPM retrieval map. Loads the cached Landsat 9
scene over the Hooghly mouth (2024-03-08), runs CBR over every water pixel,
and writes SPM concentration as both a NetCDF (for further analysis) and a
log-scaled PNG (for inspection).

Water masking: NDWI = (green - nir) / (green + nir) > 0. Simple but
effective for this scene — Sundarbans has bright mangrove canopies that
get correctly excluded.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rich.console import Console

from driftscope.spm import predict_cbr

console = Console()

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "spm" / "landsat_test.nc"
MODEL = ROOT / "data" / "spm" / "cbr_model_aquamatch_us_v1.pkl"
OUT_NC = ROOT / "data" / "spm" / "sundarbans_spm_2024-03-08.nc"
OUT_PNG = ROOT / "data" / "spm" / "sundarbans_spm_2024-03-08.png"

console.print(f"[cyan]·[/cyan] loading model {MODEL.name}")
with open(MODEL, "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"] if isinstance(bundle, dict) else bundle
console.print(
    f"  centroids {model.centroids.shape}  coefs {model.coefs.shape}  "
    f"log_target={model.log_target}  bands={bundle.get('bands') if isinstance(bundle, dict) else '?'}"
)

console.print(f"[cyan]·[/cyan] loading scene {SCENE.name}")
ds = xr.open_dataset(SCENE).load()
refl = ds.reflectance  # (band, lat, lon)
n_bands, n_lat, n_lon = refl.shape
console.print(f"  shape: {n_bands} bands × {n_lat} × {n_lon} pixels = {n_lat*n_lon:,} total")

# (band, lat, lon) → (lat*lon, band). The CBR model was trained on
# (blue, green, red, nir, swir1, swir2); the scene has the same ordering.
arr = refl.values.transpose(1, 2, 0).reshape(-1, n_bands)

# Validity mask: every band must be finite & in [0, 1].
valid = np.isfinite(arr).all(axis=1) & (arr > 0).all(axis=1) & (arr < 1).all(axis=1)
console.print(f"  {valid.sum():,} pixels with all-band data ({valid.mean()*100:.1f}%)")

# Water mask via NDWI (green - nir)/(green + nir) > 0
green = arr[:, 1]
nir = arr[:, 3]
ndwi = (green - nir) / (green + nir + 1e-9)
water = valid & (ndwi > 0.0)
console.print(f"  {water.sum():,} water pixels ({water.mean()*100:.1f}%)")

console.print(f"[cyan]·[/cyan] running predict_cbr (blend=hard) on water pixels")
spm = np.full(arr.shape[0], np.nan, dtype=np.float32)
spm[water] = predict_cbr(arr[water], model, blend="hard").astype(np.float32)

# Sanity: clip absurd predictions. Anything > 5000 mg/L is outside training
# range (max_value_mg_l filter in build_matchups). Anything < 0.1 mg/L is
# likely a numerical artifact.
extreme = water & ((spm < 0.1) | (spm > 5000.0))
console.print(f"  clipping {extreme.sum():,} extreme predictions to NaN")
spm[extreme] = np.nan

valid_spm = spm[water & np.isfinite(spm)]
console.print(
    f"[green]✓[/green] SPM map: n={len(valid_spm):,}  "
    f"median={np.median(valid_spm):.1f} mg/L  "
    f"P10={np.percentile(valid_spm, 10):.1f}  "
    f"P90={np.percentile(valid_spm, 90):.1f}  "
    f"max={valid_spm.max():.1f}"
)

# Reshape back to (lat, lon) and save NetCDF
spm_grid = spm.reshape(n_lat, n_lon)
spm_da = xr.DataArray(
    spm_grid,
    coords={"latitude": ds.latitude, "longitude": ds.longitude},
    dims=("latitude", "longitude"),
    name="spm_mg_l",
    attrs={
        "units": "mg/L",
        "long_name": "Suspended Particulate Matter (CBR retrieval)",
        "model": "CBR fit on AquaMatch v2 (US, estuary+stream, 2013-2024, k=8)",
        "source_scene": refl.attrs.get("source", ""),
        "scene_datetime": refl.attrs.get("datetime", ""),
        "water_mask": "NDWI=(green-nir)/(green+nir) > 0",
        "blend_mode": "hard",
    },
)
OUT_NC.parent.mkdir(parents=True, exist_ok=True)
spm_da.to_netcdf(OUT_NC)
console.print(f"[green]✓[/green] saved {OUT_NC.name}")

# Quick visualization — log-scale SPM with a perceptually uniform colormap.
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
vmin = max(1.0, float(np.nanpercentile(spm_grid, 1)))
vmax = min(2000.0, float(np.nanpercentile(spm_grid, 99)))
im = ax.pcolormesh(
    ds.longitude, ds.latitude, spm_grid,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis", shading="auto",
)
cb = plt.colorbar(im, ax=ax, label="SPM (mg/L, log scale)", shrink=0.7)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    f"Sundarbans SPM — {refl.attrs.get('datetime', '')[:10]}\n"
    f"Landsat 9 + CBR (AquaMatch US training, k=8, hard blend)"
)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
console.print(f"[green]✓[/green] saved {OUT_PNG.name}")

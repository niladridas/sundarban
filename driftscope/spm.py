"""Suspended Particulate Matter (SPM) retrieval from multispectral imagery.

Module 7 of the DriftScope roadmap. Unlike the physics modules (currents,
tides, waves, wind, settling) which produce velocity fields or per-particle
state, this module produces a **2D SPM concentration map** from satellite
imagery — typically used to set the *initial conditions* for a Lagrangian
release (where is the sediment plume right now?) or as a tracer field for
validation.

Architectural choice: cluster-based regression (CBR) following
**Geyman & Maloof (2019)**, *Earth and Space Science* 6, 527–537,
doi:10.1029/2018EA000539. Original target was bathymetry on the Great
Bahama Bank; we adapt the same structure to SPM retrieval. The CBR
algorithm is the right structural choice for the optically heterogeneous
GBM delta because:

  1. River-plume / mixing-zone / shelf-marine / CDOM-rich water all have
     fundamentally different SPM↔reflectance relationships.
  2. A single global regression averages across these and biases each.
  3. CBR partitions the image spectrally first (k-means on band vectors),
     fits a separate regression per class, and blends class predictions
     with weights = 1 / spectral-distance to each class centroid.
  4. Compared to SVM (the natural ML alternative), CBR extrapolates
     gracefully when calibration data are sparse, because each per-class
     model is still log-linear in physically meaningful variables.

This file currently contains the CBR algorithm (pure numpy + sklearn) and
a smoke test on synthetic data that demonstrates CBR reconstructing two
known water-type relationships. The Sentinel-2 fetch and in-situ matchup
ingest are stubs — those depend on credentials/data the project doesn't
yet have.

References
----------
Geyman & Maloof (2019)            — CBR algorithm
Tian et al. (2026), Nat Geo       — pan-Arctic SSC retrieval framework
Nechad et al. (2010)              — single-band semi-analytical SPM
Dethier et al. (2022), Science    — global SPM dataset
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.console import Console

console = Console()


# ── Spectral feature engineering ─────────────────────────────────────────────
# CBR works on a feature vector per pixel: typically log(band) and log(band
# ratios). The Geyman paper uses 5 ratios for bathymetry; the Tian SSC paper
# uses 9 spectral predictors selected via forward feature selection. Both
# patterns are captured by `band_ratio_features`, which generates all unique
# log-ratio pairs from a band set, plus the log of each band.


def band_ratio_features(
    bands: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build log-ratio + log-band feature vectors from multispectral pixels.

    Parameters
    ----------
    bands
        Pixel band reflectances, shape (n_pixels, n_bands). Values should
        be in (0, 1] — water-leaving reflectance after atmospheric correction.
    eps
        Small floor added to bands before log to avoid -inf for very dark
        pixels (the Geyman paper does the same; values << eps are usually
        cloud shadow or land mask artifacts that won't survive QA anyway).

    Returns
    -------
    features
        Shape (n_pixels, n_bands + n_bands*(n_bands-1)). The first n_bands
        columns are log(band_i); the remainder are log(band_i)/log(band_j)
        for all i ≠ j (matching Stumpf et al. 2003 / Geyman 2019 form).
    """
    safe = np.clip(bands, eps, None)
    log_b = np.log(safe)
    n_bands = bands.shape[1]

    feats = [log_b]
    for i in range(n_bands):
        for j in range(n_bands):
            if i == j:
                continue
            denom = np.where(np.abs(log_b[:, j]) < eps, eps, log_b[:, j])
            feats.append((log_b[:, i] / denom).reshape(-1, 1))
    return np.hstack(feats)


# ── CBR model ────────────────────────────────────────────────────────────────


@dataclass
class CBRModel:
    """Fitted cluster-based regression model.

    Attributes
    ----------
    centroids
        Cluster centers in spectral-feature space, shape (k, n_features).
        Used at predict time to compute per-pixel spectral distances.
    coefs
        Per-cluster regression coefficients, shape (k, n_features).
    intercepts
        Per-cluster regression intercepts, shape (k,).
    log_target
        Whether the model was fit on log(SPM) (recommended — SPM is
        log-normally distributed) and predictions need exp() before return.
    feature_eps
        Floor used in band_ratio_features; stored so predict() uses the
        same value as fit().
    """

    centroids: np.ndarray
    coefs: np.ndarray
    intercepts: np.ndarray
    log_target: bool = True
    feature_eps: float = 1e-6


def fit_cbr(
    bands: np.ndarray,
    spm: np.ndarray,
    k: int = 8,
    log_target: bool = True,
    feature_eps: float = 1e-6,
    random_state: int = 0,
) -> CBRModel:
    """Fit a CBR model from in-situ SPM ↔ reflectance matchups.

    Parameters
    ----------
    bands
        Reflectance vectors at matchup locations, shape (n, n_bands).
    spm
        In-situ SPM concentrations (mg/L), shape (n,). Strictly positive.
    k
        Number of spectral classes. Geyman uses 8; the algorithm is
        insensitive to k in [4, 12].
    log_target
        Fit log(SPM) instead of raw SPM. SPM spans 3+ orders of magnitude
        in coastal waters, so log-fit is almost always the right call.
    feature_eps
        Floor used in log() to avoid -inf at very dark pixels.
    random_state
        Seed for k-means initialization. Make this fixed in production runs.
    """
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression

    X = band_ratio_features(bands, eps=feature_eps)
    y = np.log(np.clip(spm, feature_eps, None)) if log_target else spm

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    classes = km.predict(X)

    n_features = X.shape[1]
    coefs = np.zeros((k, n_features), dtype=float)
    intercepts = np.zeros(k, dtype=float)
    for c in range(k):
        mask = classes == c
        if mask.sum() < 3:  # too few samples to fit reliably
            # fall back to global mean — this class will get zero weight
            # in practice if its centroid is far from real pixels.
            intercepts[c] = float(np.mean(y))
            continue
        lr = LinearRegression().fit(X[mask], y[mask])
        coefs[c] = lr.coef_
        intercepts[c] = lr.intercept_

    return CBRModel(
        centroids=km.cluster_centers_,
        coefs=coefs,
        intercepts=intercepts,
        log_target=log_target,
        feature_eps=feature_eps,
    )


def predict_cbr(
    bands: np.ndarray,
    model: CBRModel,
    distance_eps: float = 1e-3,
) -> np.ndarray:
    """Predict SPM at new pixels via spectral-distance-weighted blend of
    per-class regressions.

    The blending weight for class c at pixel p is:
        w_pc = 1 / (||X_p - centroid_c||² + eps)
    normalized so sum_c w_pc = 1. Following Geyman 2019, this gives smooth
    transitions between classes rather than hard cluster boundaries — the
    final estimate is *always* a weighted mix of all class models.
    """
    X = band_ratio_features(bands, eps=model.feature_eps)
    # Pairwise spectral distances pixel → centroid
    diffs = X[:, None, :] - model.centroids[None, :, :]   # (n_pixels, k, n_feat)
    dists = np.sum(diffs * diffs, axis=2)                  # (n_pixels, k)
    weights = 1.0 / (dists + distance_eps)
    weights /= weights.sum(axis=1, keepdims=True)

    per_class_pred = X @ model.coefs.T + model.intercepts[None, :]  # (n_pixels, k)
    blended = (per_class_pred * weights).sum(axis=1)

    return np.exp(blended) if model.log_target else blended


# ── Data pipeline stubs ──────────────────────────────────────────────────────
# These are deliberately not implemented yet because they depend on:
#   - Sentinel-2 access (Microsoft Planetary Computer or Copernicus DataSpace)
#   - Atmospheric correction (Sen2Cor / ACOLITE / iCOR — pick one)
#   - In-situ SPM matchups (we don't yet have a Bay-of-Bengal compilation;
#     candidates: AquaSat, GRQA, HYDRO-WEB, project-specific samples)


def fetch_sentinel2_scene(
    bbox: tuple[float, float, float, float],
    date: str,
    out_dir: Path,
    cloud_max: float = 30.0,
) -> Path:
    """Fetch a single Sentinel-2 L2A (atmospherically corrected) scene.

    NOT YET IMPLEMENTED. Recommended approach:
      1. Use `planetary-computer` + `pystac-client` to query Microsoft's
         STAC catalog at https://planetarycomputer.microsoft.com/.
      2. Filter by cloud_cover < `cloud_max`.
      3. Download bands B02 (490 nm), B03 (560 nm), B04 (665 nm),
         B05 (705 nm), B06 (740 nm), B07 (783 nm), B08 (842 nm), B8A (865 nm).
      4. Resample to a common 10 m or 20 m grid.
      5. Save as a single NetCDF with band as a dimension, matching the
         format of `data/currents/*.nc` for downstream consistency.
    """
    raise NotImplementedError(
        "Sentinel-2 fetch not implemented. Most direct path:\n"
        "  pip install planetary-computer pystac-client rioxarray\n"
        "Then query stac-api.com Microsoft Planetary Computer catalog\n"
        "for sentinel-2-l2a items intersecting bbox + date range."
    )


def load_matchups_csv(path: Path, band_columns: Sequence[str],
                      spm_column: str = "spm_mg_L") -> tuple[np.ndarray, np.ndarray]:
    """Load in-situ SPM ↔ band-reflectance matchups for CBR training.

    NOT YET IMPLEMENTED. Expected input is a CSV with columns:
      lat, lon, datetime, spm_mg_L, B02, B03, B04, B05, B06, B07, B08, B8A
    where the band columns hold the Sentinel-2 reflectance at the
    sampling site/time. Construction of such a dataset for the GBM is
    a research task in itself (see Tian et al. 2026 Methods for an
    Arctic-region equivalent that uses AquaSat + GRQA + RATS).
    """
    raise NotImplementedError(
        "Matchup ingest not implemented. See module docstring for the "
        "expected CSV schema and references."
    )


# ── Smoke test on synthetic data ─────────────────────────────────────────────
# Validates the CBR algorithm without depending on Sentinel-2 or matchups.
# Generates two distinct synthetic water types with different SPM↔reflectance
# relationships, fits CBR, and checks that recovery beats a global linear
# regression. This is the test that exercises the full algorithmic path.


def synthetic_cbr_dataset(
    n_per_type: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-water-type synthetic SPM dataset for testing CBR.

    Type A: clear shelf water — SPM driven by red band, low concentrations.
    Type B: turbid plume — SPM driven by NIR (red saturates), high concs.
    Returns (bands [4-band], spm [mg/L], type_labels).
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    # Bands ordered: blue, green, red, NIR
    # Type A — clear shelf
    spm_a = np.exp(rng.normal(np.log(5), 0.6, n_per_type))         # 5 mg/L typical
    blue_a  = 0.04 + 0.0010 * spm_a + rng.normal(0, 0.003, n_per_type)
    green_a = 0.05 + 0.0020 * spm_a + rng.normal(0, 0.003, n_per_type)
    red_a   = 0.02 + 0.0040 * spm_a + rng.normal(0, 0.003, n_per_type)
    nir_a   = 0.005 + 0.0008 * spm_a + rng.normal(0, 0.001, n_per_type)
    bands_a = np.stack([blue_a, green_a, red_a, nir_a], axis=1)

    # Type B — turbid plume
    spm_b = np.exp(rng.normal(np.log(150), 0.7, n_per_type))       # 150 mg/L typical
    blue_b  = 0.06 + 0.00010 * spm_b + rng.normal(0, 0.005, n_per_type)
    green_b = 0.10 + 0.00020 * spm_b + rng.normal(0, 0.005, n_per_type)
    red_b   = 0.13 + 0.00010 * spm_b + rng.normal(0, 0.005, n_per_type)  # saturating
    nir_b   = 0.02 + 0.00060 * spm_b + rng.normal(0, 0.003, n_per_type)
    bands_b = np.stack([blue_b, green_b, red_b, nir_b], axis=1)

    bands = np.vstack([bands_a, bands_b])
    bands = np.clip(bands, 1e-4, 1.0)  # keep in physical range
    spm = np.concatenate([spm_a, spm_b])
    labels = np.array([0]*n_per_type + [1]*n_per_type)
    perm = rng.permutation(len(spm))
    return bands[perm], spm[perm], labels[perm]


def main() -> None:
    p = argparse.ArgumentParser(description="SPM retrieval (Module 7) — CBR algorithm")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run synthetic-data validation of the CBR algorithm")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    if not args.smoke_test:
        console.print(
            "[yellow]No action.[/yellow] Module 7 currently has the CBR core "
            "and synthetic-data smoke test. Pass --smoke-test to verify the "
            "algorithm. Real-data path requires Sentinel-2 fetch and in-situ "
            "matchups (see module docstring)."
        )
        return

    rng = np.random.default_rng(0)
    bands, spm, labels = synthetic_cbr_dataset(n_per_type=300, rng=rng)
    console.print(
        f"[cyan]·[/cyan] synthetic dataset: {len(spm)} samples, "
        f"SPM range {spm.min():.1f} – {spm.max():.1f} mg/L"
    )

    # 80/20 train/test split
    perm = rng.permutation(len(spm))
    cut = int(0.8 * len(spm))
    tr, te = perm[:cut], perm[cut:]

    # Baseline: global log-linear regression (no clustering)
    from sklearn.linear_model import LinearRegression
    X_tr = band_ratio_features(bands[tr])
    X_te = band_ratio_features(bands[te])
    lr = LinearRegression().fit(X_tr, np.log(spm[tr]))
    spm_lr = np.exp(lr.predict(X_te))

    # CBR
    model = fit_cbr(bands[tr], spm[tr], k=args.k)
    spm_cbr = predict_cbr(bands[te], model)

    def metrics(true, pred):
        log_err = np.log(pred) - np.log(true)
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae_log = float(np.mean(np.abs(log_err)))
        return rmse, mae_log

    rmse_lr, mae_log_lr = metrics(spm[te], spm_lr)
    rmse_cbr, mae_log_cbr = metrics(spm[te], spm_cbr)

    console.print()
    console.print(f"[bold]Test-set performance[/bold] (n={len(te)}):")
    console.print(
        f"  global log-linear : RMSE = {rmse_lr:7.2f} mg/L  ·  "
        f"|log err| = {mae_log_lr:.3f}"
    )
    console.print(
        f"  CBR (k={args.k})         : RMSE = {rmse_cbr:7.2f} mg/L  ·  "
        f"|log err| = {mae_log_cbr:.3f}"
    )
    if rmse_cbr < rmse_lr:
        console.print(
            f"[green]✓[/green] CBR beats global by "
            f"{(1 - rmse_cbr/rmse_lr)*100:.0f}% on RMSE"
        )
    else:
        console.print(
            f"[yellow]![/yellow] CBR didn't beat global on this seed/k. "
            "Try a different k or random_state, or check feature engineering."
        )


if __name__ == "__main__":
    main()

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

    Two spaces matter, and they're different:
      - Clustering space: raw band reflectances (n_bands). This is where
        k-means runs and where spectral distances are computed at predict
        time. Keeping it low-dimensional (~4-10 bands) keeps cluster
        boundaries sharp and physically interpretable.
      - Regression space: log + log-ratio features (n_features). Per-class
        regressions live here. Higher-dimensional, but each cluster fits
        a linear model in this space — physically grounded by Beer-Lambert
        / SPM scattering relationships.
    """

    centroids: np.ndarray   # (k, n_bands)  — clustering space
    coefs: np.ndarray       # (k, n_features) — regression space
    intercepts: np.ndarray  # (k,)
    log_target: bool = True
    feature_eps: float = 1e-6


def fit_cbr(
    bands: np.ndarray,
    spm: np.ndarray,
    k: int = 8,
    log_target: bool = True,
    feature_eps: float = 1e-6,
    random_state: int = 0,
    ridge_alpha: float = 0.1,
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
    from sklearn.linear_model import Ridge

    # Cluster on raw bands (Geyman 2019, Section 2.2.3) — keeps cluster
    # boundaries spectrally meaningful instead of in derived-feature space.
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(bands)
    classes = km.predict(bands)

    # Per-class regression in feature space. Ridge instead of plain LR:
    # the per-cluster training set can be sparse (a few dozen samples for
    # a 16-feature space), so ridge regularization prevents wildly large
    # coefficients that would explode in exp() at predict time.
    X = band_ratio_features(bands, eps=feature_eps)
    y = np.log(np.clip(spm, feature_eps, None)) if log_target else spm

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
        lr = Ridge(alpha=ridge_alpha).fit(X[mask], y[mask])
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
    blend: str = "hard",
    distance_eps: float = 1e-3,
) -> np.ndarray:
    """Predict SPM at new pixels.

    Two blending modes:

    blend="hard" (default for SPM): assign each pixel to its nearest
    cluster centroid in band space, then evaluate only that cluster's
    regression. Robust when per-class SPM means differ by orders of
    magnitude (clear vs turbid water).

    blend="soft": weighted mean of all class predictions using inverse
    spectral distance (Geyman 2019 form). Good when all classes predict
    targets in similar ranges (their bathymetry case: all classes predict
    depths in 0–3 m). For SPM, soft blending lets wrong-cluster regressions
    contaminate predictions by their mean — a 15% weight on a turbid-water
    model evaluated at a clear-water pixel can shift predictions by
    hundreds of mg/L.

    Distances are always computed in the clustering space (raw bands),
    not in the higher-dimensional feature space.
    """
    X = band_ratio_features(bands, eps=model.feature_eps)
    diffs = bands[:, None, :] - model.centroids[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))             # (n_pixels, k) L2

    per_class_pred = X @ model.coefs.T + model.intercepts[None, :]

    if blend == "hard":
        cluster = np.argmin(dists, axis=1)
        rows = np.arange(len(bands))
        result = per_class_pred[rows, cluster]
    elif blend == "soft":
        weights = 1.0 / (dists + distance_eps)
        weights /= weights.sum(axis=1, keepdims=True)
        result = (per_class_pred * weights).sum(axis=1)
    else:
        raise ValueError(f"blend must be 'hard' or 'soft', got {blend!r}")

    return np.exp(result) if model.log_target else result


# ── Data pipeline stubs ──────────────────────────────────────────────────────
# These are deliberately not implemented yet because they depend on:
#   - Sentinel-2 access (Microsoft Planetary Computer or Copernicus DataSpace)
#   - Atmospheric correction (Sen2Cor / ACOLITE / iCOR — pick one)
#   - In-situ SPM matchups (we don't yet have a Bay-of-Bengal compilation;
#     candidates: AquaSat, GRQA, HYDRO-WEB, project-specific samples)


SENTINEL2_DEFAULT_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
S2_BAND_WAVELENGTHS_NM = {
    "B02": 490, "B03": 560, "B04": 665, "B05": 705,
    "B06": 740, "B07": 783, "B08": 842, "B8A": 865,
}

# Microsoft Planetary Computer and Element84 host the same Sentinel-2 L2A
# data but use different asset key conventions. Map our canonical band IDs
# to each catalog's asset names.
S2_BAND_ALIASES = {
    "B02": {"element84": "blue",     "pc": "B02"},
    "B03": {"element84": "green",    "pc": "B03"},
    "B04": {"element84": "red",      "pc": "B04"},
    "B05": {"element84": "rededge1", "pc": "B05"},
    "B06": {"element84": "rededge2", "pc": "B06"},
    "B07": {"element84": "rededge3", "pc": "B07"},
    "B08": {"element84": "nir",      "pc": "B08"},
    "B8A": {"element84": "nir08",    "pc": "B8A"},
}
S2_CATALOG_URLS = {
    "element84": "https://earth-search.aws.element84.com/v1/",
    "pc": "https://planetarycomputer.microsoft.com/api/stac/v1",
}


def fetch_sentinel2_scene(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    out_path: Path,
    cloud_max: float = 30.0,
    bands: Sequence[str] = SENTINEL2_DEFAULT_BANDS,
    resolution_m: int = 20,
    catalog: str = "element84",
    force: bool = False,
) -> Path:
    """Fetch the least-cloudy Sentinel-2 L2A scene for bbox + date range.

    Saves a single NetCDF with band as a dimension, lat/lon coords (EPSG:4326),
    and reflectance values in [0, 1] (Sentinel-2 L2A scaled by 1/10000).

    Parameters
    ----------
    bbox
        (lon_min, lat_min, lon_max, lat_max), EPSG:4326.
    start_date, end_date
        ISO date strings, e.g. '2024-03-15' / '2024-03-30'.
    out_path
        Output NetCDF path.
    cloud_max
        Max scene-level cloud cover percentage (0–100). 30 is permissive;
        try 10 for the cleanest product, accepting fewer matches.
    bands
        Sentinel-2 band IDs. Default 8 visible+NIR bands.
    resolution_m
        Output resolution in meters; bands are upsampled/downsampled to match.
        20 is the natural common grid (B05/B06/B07/B8A native, others 10m).
    force
        Re-download even if out_path exists.
    """
    if out_path.exists() and not force:
        console.print(f"[green]✓[/green] cached: {out_path.name}")
        return out_path

    if catalog not in S2_CATALOG_URLS:
        raise ValueError(
            f"catalog must be one of {list(S2_CATALOG_URLS)}; got {catalog!r}"
        )

    import pystac_client
    from odc.stac import load as odc_load

    if catalog == "pc":
        import planetary_computer as pc
        client = pystac_client.Client.open(
            S2_CATALOG_URLS["pc"], modifier=pc.sign_inplace
        )
    else:
        client = pystac_client.Client.open(S2_CATALOG_URLS[catalog])

    # Translate canonical band IDs to this catalog's asset names.
    asset_names = [S2_BAND_ALIASES[b][catalog] for b in bands]

    console.print(
        f"[cyan]↓[/cyan] querying {catalog} for Sentinel-2 L2A "
        f"bbox={bbox} dates={start_date}/{end_date} cloud<{cloud_max}%"
    )
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes found in {bbox} for "
            f"{start_date}/{end_date} with cloud_cover<{cloud_max}%. "
            f"Try widening the date range or raising cloud_max."
        )
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    best = items[0]
    best_date = best.datetime.date()
    # Take all tiles from the same overpass day. Sentinel-2 splits the
    # world into MGRS tiles; a single bbox often spans 2-4 of them, and
    # odc-stac will mosaic them as long as we pass them all.
    day_items = [i for i in items if i.datetime.date() == best_date]
    avg_cc = sum(i.properties["eo:cloud_cover"] for i in day_items) / len(day_items)
    console.print(
        f"[cyan]·[/cyan] {len(items)} scenes match; using "
        f"{len(day_items)} tile(s) from {best_date} "
        f"(avg {avg_cc:.1f}% cloud)"
    )

    resolution_deg = resolution_m / 111_000.0
    ds = odc_load(
        day_items,
        bands=asset_names,
        bbox=list(bbox),
        crs="EPSG:4326",
        resolution=resolution_deg,
        chunks={},
    )
    # Rename the asset-named variables back to canonical band IDs so the
    # output NetCDF is catalog-agnostic.
    rename_map = {S2_BAND_ALIASES[b][catalog]: b for b in bands}
    ds = ds.rename(rename_map)
    # ds has shape (time, y, x) per band. Drop time (single scene).
    ds = ds.isel(time=0)

    # Sentinel-2 L2A surface reflectance is stored as DN in [0, 10000].
    # The masked nodata value (0) becomes NaN after reflectance scaling.
    # odc-stac with crs="EPSG:4326" produces latitude/longitude coords natively.
    refl = (ds.where(ds > 0).to_array(dim="band").astype("float32")) / 10000.0
    refl.name = "reflectance"
    refl.attrs.update({
        "units": "1",
        "long_name": "Surface reflectance, Sentinel-2 L2A",
        "source": f"sentinel-2-l2a/{','.join(i.id for i in day_items)}",
        "datetime": best.datetime.isoformat(),
        "cloud_cover_pct": float(avg_cc),
        "wavelengths_nm": ",".join(
            f"{b}={S2_BAND_WAVELENGTHS_NM.get(b, '?')}" for b in bands
        ),
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    refl.to_dataset().to_netcdf(out_path)
    console.print(
        f"[green]✓[/green] saved {out_path.name} "
        f"({refl.sizes['latitude']}×{refl.sizes['longitude']}, "
        f"{len(bands)} bands)"
    )
    return out_path


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
    """Two-water-type synthetic SPM dataset designed to expose CBR's advantage.

    Type A — clear shelf water: SPM is well-characterized by visible bands;
    NIR is barely above zero (Type-A water is too clear for NIR to track SPM).

    Type B — turbid plume: red band SATURATES at high SPM (the Nechad-2010
    well-known optics problem: water-leaving reflectance plateaus near 0.18
    once SPM exceeds ~200 mg/L). NIR remains a good SPM proxy because it's
    not yet saturating. This is the exact regime where Tian et al. 2026's
    global RF model breaks down (>500 mg/L) and where a single global
    log-linear regression cannot reconcile the two regimes.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    # Bands ordered: blue, green, red, NIR
    # Type A — clear shelf, linear bands ~ SPM
    spm_a = np.exp(rng.normal(np.log(5), 0.6, n_per_type))
    blue_a  = 0.04 + 0.0010 * spm_a + rng.normal(0, 0.003, n_per_type)
    green_a = 0.05 + 0.0020 * spm_a + rng.normal(0, 0.003, n_per_type)
    red_a   = 0.02 + 0.0040 * spm_a + rng.normal(0, 0.003, n_per_type)
    nir_a   = 0.005 + 0.00010 * spm_a + rng.normal(0, 0.001, n_per_type)
    bands_a = np.stack([blue_a, green_a, red_a, nir_a], axis=1)

    # Type B — turbid plume; red SATURATES at high SPM (tanh form), NIR linear
    spm_b = np.exp(rng.normal(np.log(300), 0.6, n_per_type))
    blue_b  = 0.06 + 0.040 * np.tanh(spm_b / 100) + rng.normal(0, 0.005, n_per_type)
    green_b = 0.08 + 0.080 * np.tanh(spm_b / 100) + rng.normal(0, 0.005, n_per_type)
    red_b   = 0.10 + 0.090 * np.tanh(spm_b / 80)  + rng.normal(0, 0.005, n_per_type)
    nir_b   = 0.02 + 0.00060 * spm_b              + rng.normal(0, 0.003, n_per_type)
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
    p.add_argument("--fetch", action="store_true",
                   help="Fetch a Sentinel-2 L2A scene to test the pipeline")
    p.add_argument("--bbox", type=str, default="88.5,21.3,89.5,22.0",
                   help="lon_min,lat_min,lon_max,lat_max (default: Hooghly mouth)")
    p.add_argument("--start", type=str, default="2024-03-01",
                   help="ISO start date (default: 2024-03-01, dry season pre-monsoon)")
    p.add_argument("--end", type=str, default="2024-03-31",
                   help="ISO end date")
    p.add_argument("--cloud-max", type=float, default=10.0,
                   help="Max scene cloud cover %% (default 10)")
    p.add_argument("--catalog", default="element84",
                   choices=list(S2_CATALOG_URLS),
                   help="STAC catalog: 'element84' (default, AWS Open Data, no auth) "
                        "or 'pc' (Microsoft Planetary Computer, frequently slow)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output NetCDF path (default: data/spm/<auto>.nc)")
    args = p.parse_args()

    if args.fetch:
        from .config import DATA
        bbox = tuple(float(x) for x in args.bbox.split(","))
        if len(bbox) != 4:
            raise SystemExit("--bbox must be 'lon_min,lat_min,lon_max,lat_max'")
        out = args.out or (DATA / "spm" /
                           f"sentinel2_{args.start}_{args.end}.nc")
        fetch_sentinel2_scene(bbox, args.start, args.end, out,
                              cloud_max=args.cloud_max,
                              catalog=args.catalog)
        return

    if not args.smoke_test:
        console.print(
            "[yellow]No action.[/yellow] Module 7 has:\n"
            "  --smoke-test    validate CBR on synthetic data\n"
            "  --fetch         download a Sentinel-2 L2A scene\n"
            "Real-data CBR training still needs in-situ SPM matchups."
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

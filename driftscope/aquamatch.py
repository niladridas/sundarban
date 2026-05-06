"""AquaMatch ingest — join AquaSat-v2 TSS measurements with SiteSR Landsat reflectances.

AquaMatch v2 (Brousil, Meyer, Willi, Steele, De La Torre, Ross 2024+) is the
pipeline-decomposed successor to AquaSat (Ross et al. 2019). The two data
packages we use:

  - **AquaMatch Total Suspended Solids** (EDI edi.2048.2): in-situ TSS/SSC
    measurements from the EPA Water Quality Portal, harmonized to mg/L.
    File: tss_harmonized_final.csv (~5M rows, 1 GB).
  - **SiteSR** (EDI edi.2254.1): Landsat C2 L2 surface reflectance at every
    WQP/NWIS site for 1983–2024. Per-platform .feather files. Smallest
    metadata CSV maps `(org_id, loc_id) → siteSR_id`. Total 48 GB unpacked.

The two are designed to be joined by the user. Join key: `(org_id, loc_id)` on
the TSS side maps to those same fields in SiteSR's site lookup, which yields
`siteSR_id`. Then `siteSR_id + date` finds the matching Landsat reflectance
row in the per-platform feather files.

Geographic note: AquaMatch is **strictly US** (EPA WQP + USGS NWIS). For Bay
of Bengal applications, you train CBR on US matchups and apply to Sundarbans
Landsat scenes — same domain-transfer pattern Tian et al. 2026 used for the
pan-Arctic. Filter to estuarine sites (~106k available) for the most
optically-similar training subset.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


# Reflectance band columns in the SiteSR Landsat feather files. We use the
# "median" stat (per-pixel median of the n×n window around the site) — it's
# more robust to single-pixel outliers than the mean. The trailing names
# match SiteSR's own naming conventions.
SITESR_BANDS = ("med_Blue", "med_Green", "med_Red", "med_Nir",
                "med_Swir1", "med_Swir2")
SITESR_BAND_RENAME = {
    "med_Blue": "blue",
    "med_Green": "green",
    "med_Red": "red",
    "med_Nir": "nir",
    "med_Swir1": "swir1",
    "med_Swir2": "swir2",
}


@dataclass
class AquaMatchPaths:
    """Paths to the AquaMatch + SiteSR data files.

    Pass to `build_matchups` after extracting the two EDI packages.
    """
    tss_csv: Path                      # tss_harmonized_final.csv
    sites_csv: Path                    # siteSR_collated_WQP_NWIS_sites_with_NHD_info_*.csv
    landsat_feathers: Sequence[Path]   # one or more siteSR_Landsat{4,5,7,8,9}_DSWE1_*.feather


def load_filtered_tss(
    tss_csv: Path,
    site_types: Sequence[str] | None = None,
    min_year: int | None = 2013,   # post-Landsat-8 era by default
    max_value_mg_l: float = 5000.0,
    drop_flagged: bool = True,
) -> pd.DataFrame:
    """Load and filter the AquaMatch TSS harmonized CSV.

    Defaults are conservative for CBR training:
      - 2013+ only, so reflectances are from Landsat 8/9 (modern OLI
        sensor, closest band correspondence to Sentinel-2).
      - Drop values >5000 mg/L (above the optical-saturation regime where
        Stokes' law fails and reflectance models break down — see Tian
        et al. 2026 caveat).
      - Drop QA-flagged rows by default.

    Returns a DataFrame with columns:
        org_id, loc_id, date, ssc_mg_l, lat, lon
    """
    console.print(f"[cyan]·[/cyan] loading {tss_csv.name}")
    cols = [
        "OrganizationIdentifier", "MonitoringLocationIdentifier",
        "ResolvedMonitoringLocationTypeName",
        "ActivityStartDate", "harmonized_value", "harmonized_units",
        "lat", "lon",
        "depth_flag", "mdl_flag", "approx_flag", "greater_flag",
        "field_flag", "misc_flag",
    ]
    df = pd.read_csv(
        tss_csv, usecols=cols,
        dtype={"OrganizationIdentifier": str,
               "MonitoringLocationIdentifier": str},
        low_memory=False,
    )
    n0 = len(df)
    df = df[df["harmonized_units"] == "mg/L"]
    df = df[(df["harmonized_value"] > 0) &
            (df["harmonized_value"] <= max_value_mg_l)]

    if drop_flagged:
        flag_cols = ["depth_flag", "mdl_flag", "approx_flag",
                     "greater_flag", "field_flag", "misc_flag"]
        # Keep rows where every flag column is 0 / NaN / empty
        for c in flag_cols:
            df = df[df[c].isna() | (df[c].astype(str).isin(["0", "0.0", "", "nan"]))]

    df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
    df = df.dropna(subset=["ActivityStartDate"])
    if min_year is not None:
        df = df[df["ActivityStartDate"].dt.year >= min_year]

    if site_types is not None:
        df = df[df["ResolvedMonitoringLocationTypeName"].isin(site_types)]

    df = df.rename(columns={
        "OrganizationIdentifier": "org_id",
        "MonitoringLocationIdentifier": "loc_id",
        "ActivityStartDate": "date",
        "harmonized_value": "ssc_mg_l",
    })
    df = df[["org_id", "loc_id", "date", "ssc_mg_l", "lat", "lon"]].reset_index(drop=True)
    console.print(
        f"[green]✓[/green] {len(df):,} TSS records after filtering "
        f"({n0:,} → {len(df):,}, dropped {(1-len(df)/n0)*100:.1f}%)"
    )
    return df


def load_site_lookup(sites_csv: Path) -> pd.DataFrame:
    """Load the (org_id, loc_id) → siteSR_id lookup table from SiteSR."""
    console.print(f"[cyan]·[/cyan] loading {sites_csv.name}")
    sites = pd.read_csv(
        sites_csv,
        usecols=["siteSR_id", "org_id", "loc_id", "harmonized_site_type",
                 "WGS84_Latitude", "WGS84_Longitude"],
        dtype={"siteSR_id": str, "org_id": str, "loc_id": str},
        low_memory=False,
    )
    console.print(f"[green]✓[/green] {len(sites):,} sites in lookup")
    return sites


def load_landsat_reflectances(
    feather_paths: Sequence[Path],
    siteSR_ids: set[str] | None = None,
    drop_clouds: bool = True,
    cloud_max: float = 0.2,
) -> pd.DataFrame:
    """Load Landsat reflectances from one or more SiteSR feather files.

    `siteSR_ids` lets you pre-filter to only sites that have TSS measurements
    (massive memory savings). Pass `None` to load everything.
    """
    import pyarrow.feather as feather
    frames = []
    for p in feather_paths:
        console.print(f"[cyan]·[/cyan] reading {p.name}")
        df = feather.read_feather(p)
        if siteSR_ids is not None:
            df = df[df["siteSR_id"].isin(siteSR_ids)]
        if drop_clouds:
            df = df[df["prop_clouds"] <= cloud_max]
        # Keep only columns we need
        keep = ["siteSR_id", "date", "mission", "prop_clouds", *SITESR_BANDS]
        df = df[keep]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", *SITESR_BANDS])
        # Drop rows where any reflectance is < 0 or > 1 (sensor artifacts)
        for b in SITESR_BANDS:
            df = df[(df[b] > 0) & (df[b] < 1.0)]
        frames.append(df)
        console.print(f"  → {len(df):,} valid scenes")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns=SITESR_BAND_RENAME)
    return out


def build_matchups(
    paths: AquaMatchPaths,
    out_csv: Path,
    site_types: Sequence[str] | None = ("Estuary", "Stream"),
    min_year: int = 2013,
    max_value_mg_l: float = 5000.0,
    time_window_days: int = 1,
    cloud_max: float = 0.2,
) -> Path:
    """Full AquaMatch → CBR-ready matchup pipeline.

    Steps:
      1. Filter the harmonized TSS to a sensible subset (estuary + stream
         by default, post-2013, value < 5000 mg/L, no QA flags).
      2. Map each TSS row's (org_id, loc_id) to siteSR_id via the SiteSR
         sites lookup.
      3. Load Landsat reflectances for those siteSR_ids only.
      4. For each TSS row, find the closest Landsat scene at the same
         site within ±`time_window_days` days. Drop rows with no match.
      5. Write a CSV with columns ready for `driftscope.spm.fit_cbr`:
           lat, lon, date, ssc_mg_l, blue, green, red, nir, swir1, swir2
    """
    tss = load_filtered_tss(
        paths.tss_csv, site_types=site_types,
        min_year=min_year, max_value_mg_l=max_value_mg_l,
    )
    sites = load_site_lookup(paths.sites_csv)

    # Map TSS to siteSR_id
    tss = tss.merge(
        sites[["siteSR_id", "org_id", "loc_id", "harmonized_site_type"]],
        on=["org_id", "loc_id"], how="inner",
    )
    console.print(
        f"[green]✓[/green] {len(tss):,} TSS records mapped to siteSR_id"
    )

    # Load Landsat reflectances ONLY for the matched site set
    rs = load_landsat_reflectances(
        paths.landsat_feathers,
        siteSR_ids=set(tss["siteSR_id"].unique()),
        cloud_max=cloud_max,
    )
    console.print(f"[green]✓[/green] {len(rs):,} Landsat scenes loaded")

    # Time-tolerant inner join: for each TSS row, find the closest scene at
    # the same siteSR_id within ±time_window_days. Use merge_asof which is
    # order-of-magnitude faster than per-row lookup for ~M-row tables.
    # merge_asof requires identical datetime dtypes on both sides
    tss["date"] = tss["date"].astype("datetime64[ns]")
    rs["date"] = rs["date"].astype("datetime64[ns]")
    tss = tss.sort_values("date").reset_index(drop=True)
    rs = rs.sort_values("date").reset_index(drop=True)
    tol = pd.Timedelta(days=time_window_days)
    matched = pd.merge_asof(
        tss, rs, on="date", by="siteSR_id",
        tolerance=tol, direction="nearest",
    )
    matched = matched.dropna(subset=["blue", "green", "red"])
    console.print(
        f"[green]✓[/green] {len(matched):,} matchups within ±{time_window_days}d "
        f"({len(matched)/len(tss)*100:.1f}% of TSS records)"
    )

    out_cols = ["lat", "lon", "date", "ssc_mg_l",
                "blue", "green", "red", "nir", "swir1", "swir2",
                "harmonized_site_type", "mission", "prop_clouds"]
    matched = matched[out_cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_csv, index=False)
    console.print(f"[green]✓[/green] saved {out_csv} ({len(matched):,} rows)")
    return out_csv


def matchups_to_cbr_input(
    matchup_csv: Path,
    band_columns: Sequence[str] = ("blue", "green", "red", "nir", "swir1", "swir2"),
    spm_column: str = "ssc_mg_l",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a matchup CSV produced by `build_matchups` into (bands, spm)
    arrays ready for `driftscope.spm.fit_cbr`."""
    df = pd.read_csv(matchup_csv)
    df = df.dropna(subset=[*band_columns, spm_column])
    df = df[df[spm_column] > 0]
    bands = df[list(band_columns)].to_numpy(dtype=float)
    spm = df[spm_column].to_numpy(dtype=float)
    return bands, spm

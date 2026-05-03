"""DriftScope dashboard.

Run: `streamlit run app.py`

Three tabs:
1. **Drift** — pick a bbox, seed point on a map, run a sim, animate trajectories
2. **Accumulation** — heatmap of where particles end up
3. **Currents** — the underlying CMEMS field that's driving everything

Designed to feel like a real product: dark theme, sensible defaults, fast iteration.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import xarray as xr

from driftscope.config import (
    CURRENTS_DIR,
    PRESETS,
    SimConfig,
    TRAJ_DIR,
    WAVES_DIR,
)
from driftscope.fetch import fetch_currents, output_path
from driftscope.simulate import run_simulation
from driftscope.stokes import fetch_stokes, output_path as stokes_output_path
from driftscope.viz import accumulation_grid, load_trajectories, traj_to_dataframe


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriftScope — Lagrangian ocean drift",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; max-width: 1400px; }
        h1 { font-weight: 700; letter-spacing: -0.02em; color: #0f172a; }
        h1 em { color: #0ea5e9; font-style: normal; font-weight: 600; }
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        .stMetric {
            background: #ffffff;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .stMetric [data-testid="stMetricLabel"] {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: #e0f2fe;
            color: #0c4a6e;
        }
        button[kind="primary"] {
            background: #0ea5e9;
            border-color: #0ea5e9;
        }
        button[kind="primary"]:hover {
            background: #0284c7;
            border-color: #0284c7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar — region & simulation controls ───────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 DriftScope")
    st.caption("Lagrangian particle tracking on satellite-derived currents")

    map_theme = st.radio(
        "Map theme",
        ["Light", "Dark"],
        horizontal=True,
        index=0,
        key="map_theme",
        help="Controls the basemap and chart background — page chrome stays light.",
    )
    st.divider()

    st.markdown("**Region**")
    preset = st.selectbox(
        "Preset",
        options=list(PRESETS.keys()) + ["custom"],
        format_func=lambda k: PRESETS[k].name if k in PRESETS else "Custom bbox",
        index=0,
        key="preset",
    )

    if preset == "custom":
        c1, c2 = st.columns(2)
        with c1:
            lon_min = st.number_input("lon min", value=88.0, format="%.2f")
            lat_min = st.number_input("lat min", value=21.0, format="%.2f")
        with c2:
            lon_max = st.number_input("lon max", value=90.5, format="%.2f")
            lat_max = st.number_input("lat max", value=22.8, format="%.2f")
        region_name = "custom"
        default_seed_lon = (lon_min + lon_max) / 2
        default_seed_lat = (lat_min + lat_max) / 2
    else:
        r = PRESETS[preset]
        lon_min, lat_min, lon_max, lat_max = r.bbox
        region_name = r.name
        default_seed_lon, default_seed_lat = r.default_seed

    st.divider()
    st.markdown("**Time window**")
    end_default = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    start_default = end_default - timedelta(days=14)
    start_date = st.date_input("Start", start_default)
    end_date = st.date_input("End", end_default)
    st.caption(
        "💡 Validation tip: monsoon BoB drift is well-documented for "
        "**Aug 2024**. Try `2024-08-01 → 2024-08-15` with Stokes on."
    )

    st.divider()
    st.markdown("**Simulation**")
    n_particles = st.slider("Particles", 50, 5000, 500, step=50)
    runtime_days = st.slider("Runtime (days)", 1, 30, 10)

    seed_mode_label = st.radio(
        "Seed mode",
        ["Point release", "Basin fill"],
        index=0,
        horizontal=True,
        help=(
            "**Point release**: seed a small disk at one location (oil-spill scenario). "
            "**Basin fill**: spread particles across every ocean cell of the bbox "
            "(see how the whole basin moves)."
        ),
    )
    seed_mode = "bbox" if seed_mode_label == "Basin fill" else "disk"

    if seed_mode == "disk":
        seed_radius = st.slider("Seed radius (°)", 0.05, 3.0, 0.15, step=0.05)
        seed_lon = st.number_input(
            "Seed lon", value=default_seed_lon, format="%.3f",
            help="Center of the particle release disk",
            key=f"seed_lon_{preset}",
        )
        seed_lat = st.number_input(
            "Seed lat", value=default_seed_lat, format="%.3f",
            key=f"seed_lat_{preset}",
        )
    else:
        seed_radius = 0.15
        seed_lon = default_seed_lon
        seed_lat = default_seed_lat
        st.caption(
            "🌐 Particles will fill every ocean cell in the bbox. "
            "Disk inputs are ignored in this mode."
        )

    st.divider()
    st.markdown("**Forcings**")
    st.caption("✓ CMEMS surface currents *(always on)*")
    include_stokes = st.toggle(
        "Stokes drift (ERA5 waves)",
        value=False,
        help=(
            "Adds surface Stokes drift derived from ERA5 wave height/period/direction "
            "to the currents. Captures wave-induced drift of floating debris. "
            "Requires a CDS API key — see .env.example."
        ),
    )
    if include_stokes:
        stokes_scale = st.slider(
            "Stokes scale", 0.5, 8.0, 1.0, step=0.5,
            help=(
                "Multiplier on the Phillips-formula Stokes magnitude. "
                "1.0 = textbook narrow-band baseline; ~8.0 matches OpenDrift's "
                "stronger convention. Use to A/B test sensitivity."
            ),
        )
    else:
        stokes_scale = 1.0

    st.divider()
    fetch_btn = st.button("⬇️  Fetch currents", width="stretch")
    fetch_waves_btn = st.button(
        "🌊  Fetch waves (ERA5)", width="stretch",
        disabled=not include_stokes,
        help="Enabled when Stokes drift is on.",
    )
    sim_btn = st.button("▶️  Run simulation", type="primary", width="stretch")
    compare_btn = st.button(
        "🔬  Run A/B (off vs on)", width="stretch",
        disabled=not include_stokes,
        help=(
            "Runs two simulations back-to-back — currents-only and "
            "currents+Stokes — and overlays the trajectories so you can "
            "see exactly how much Stokes shifts particles."
        ),
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"# DriftScope · _{region_name}_"
)
st.caption(
    f"bbox  ·  {lon_min:.2f}, {lat_min:.2f}  →  {lon_max:.2f}, {lat_max:.2f}      "
    f"·   {start_date} → {end_date}"
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _bbox_view_state(lon_min, lat_min, lon_max, lat_max, zoom_pad=0.0) -> pdk.ViewState:
    return pdk.ViewState(
        longitude=(lon_min + lon_max) / 2,
        latitude=(lat_min + lat_max) / 2,
        zoom=_estimate_zoom(lon_max - lon_min, lat_max - lat_min) + zoom_pad,
        pitch=0,
        bearing=0,
    )


def _estimate_zoom(dlon: float, dlat: float) -> float:
    span = max(dlon, dlat)
    if span < 0.5:
        return 9.5
    if span < 1.5:
        return 8.0
    if span < 5:
        return 6.5
    if span < 12:
        return 5.0
    return 3.5


@st.cache_data(show_spinner=False)
def _load_currents_meta(path_str: str) -> dict:
    ds = xr.open_dataset(path_str)
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    return {
        "n_times": int(ds.sizes.get("time", 0)),
        "n_lon": int(ds.sizes[lon_name]),
        "n_lat": int(ds.sizes[lat_name]),
        "t0": str(ds.time.values[0])[:10],
        "t1": str(ds.time.values[-1])[:10],
        "u_max": float(np.nanmax(np.abs(ds.uo.values))),
    }


def _latest_traj() -> Path | None:
    files = sorted(TRAJ_DIR.glob("*_traj.zarr"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ── Action handlers ──────────────────────────────────────────────────────────
bbox = (lon_min, lat_min, lon_max, lat_max)
currents_path = output_path(region_name, str(start_date), str(end_date))
waves_path = stokes_output_path(region_name, str(start_date), str(end_date))

if fetch_btn:
    with st.spinner(f"Downloading CMEMS currents for {region_name}…"):
        try:
            fetch_currents(
                bbox=bbox,
                start_date=str(start_date),
                end_date=str(end_date),
                region_name=region_name,
            )
            st.success(f"✓ Saved {currents_path.name}")
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.info(
                "Make sure CMEMS credentials are set in `.env` "
                "(free signup at https://marine.copernicus.eu)."
            )

if fetch_waves_btn:
    with st.spinner(f"Downloading ERA5 waves for {region_name} (CDS queue can take a few min)…"):
        try:
            fetch_stokes(
                bbox=bbox,
                start_date=str(start_date),
                end_date=str(end_date),
                region_name=region_name,
            )
            st.success(f"✓ Saved {waves_path.name}")
        except Exception as e:
            st.error(f"Wave fetch failed: {e}")
            st.info(
                "Make sure CDS API credentials are set: either CDSAPI_URL + "
                "CDSAPI_KEY in `.env`, or a `~/.cdsapirc` file. "
                "Free key: https://cds.climate.copernicus.eu/api-how-to"
            )

def _ensure_currents():
    """Resolve the currents NetCDF or stop with an error message."""
    global currents_path
    if currents_path.exists():
        return currents_path
    existing = sorted(CURRENTS_DIR.glob("*.nc"), key=lambda p: p.stat().st_mtime)
    if existing:
        st.info(f"Using cached currents: {existing[-1].name}")
        return existing[-1]
    st.error("No currents file. Click ⬇️ Fetch currents first.")
    st.stop()


def _ensure_waves():
    """Resolve the waves NetCDF or stop with an error message."""
    global waves_path
    if waves_path.exists():
        return waves_path
    existing = sorted(WAVES_DIR.glob("*.nc"), key=lambda p: p.stat().st_mtime)
    if existing:
        st.info(f"Using cached waves: {existing[-1].name}")
        return existing[-1]
    st.error("Stokes is enabled but no waves file exists. Click 🌊 Fetch waves first.")
    st.stop()


def _stokes_shift_km(off_path: Path, on_path: Path) -> tuple[float, float, int]:
    """Per-particle great-circle distance between matching trajectory endpoints."""
    df_off = traj_to_dataframe(load_trajectories(off_path))
    df_on = traj_to_dataframe(load_trajectories(on_path))
    end_off = df_off.sort_values("time").groupby("traj").tail(1).set_index("traj")
    end_on = df_on.sort_values("time").groupby("traj").tail(1).set_index("traj")
    common = end_off.index.intersection(end_on.index)
    if len(common) == 0:
        return 0.0, 0.0, 0
    dlon = (end_on.loc[common, "lon"] - end_off.loc[common, "lon"]) * np.cos(
        np.radians(end_off.loc[common, "lat"])
    )
    dlat = end_on.loc[common, "lat"] - end_off.loc[common, "lat"]
    dist_km = np.sqrt(dlon ** 2 + dlat ** 2) * 111.0
    return float(dist_km.mean()), float(dist_km.median()), len(common)


if sim_btn:
    # Clear any prior A/B state so the Drift tab doesn't overlay stale runs.
    for k in ("ab_mode", "ab_off_path", "ab_on_path"):
        st.session_state.pop(k, None)

    currents_path = _ensure_currents()
    stokes_arg = _ensure_waves() if include_stokes else None

    cfg = SimConfig(
        n_particles=n_particles,
        runtime_days=runtime_days,
        seed_lon=seed_lon,
        seed_lat=seed_lat,
        seed_radius_deg=seed_radius,
        seed_mode=seed_mode,
        include_stokes=include_stokes,
        stokes_scale=stokes_scale,
    )
    with st.spinner(f"Advecting {n_particles} particles for {runtime_days} days…"):
        try:
            traj_path = run_simulation(currents_path, cfg, stokes_nc=stokes_arg)
            st.success(f"✓ Trajectories: {traj_path.name}")
            st.session_state["traj_path"] = str(traj_path)
        except Exception as e:
            st.exception(e)


if compare_btn:
    currents_path = _ensure_currents()
    waves_path = _ensure_waves()

    base_cfg = dict(
        n_particles=n_particles,
        runtime_days=runtime_days,
        seed_lon=seed_lon,
        seed_lat=seed_lat,
        seed_radius_deg=seed_radius,
        seed_mode=seed_mode,
    )
    cfg_off = SimConfig(**base_cfg, include_stokes=False, stokes_scale=1.0)
    cfg_on = SimConfig(**base_cfg, include_stokes=True, stokes_scale=stokes_scale)

    try:
        with st.spinner("A/B run 1/2: currents only…"):
            traj_off = run_simulation(currents_path, cfg_off)
        with st.spinner(f"A/B run 2/2: currents + Stokes (×{stokes_scale:.1f})…"):
            traj_on = run_simulation(currents_path, cfg_on, stokes_nc=waves_path)

        mean_km, med_km, n = _stokes_shift_km(traj_off, traj_on)
        st.success(
            f"✓ A/B done. Stokes-induced endpoint shift across {n} particles: "
            f"**mean {mean_km:.2f} km · median {med_km:.2f} km**"
        )

        st.session_state["ab_mode"] = True
        st.session_state["ab_off_path"] = str(traj_off)
        st.session_state["ab_on_path"] = str(traj_on)
        # Also point the single-run state at the +Stokes run so other tabs
        # (Accumulation, etc.) have something to render.
        st.session_state["traj_path"] = str(traj_on)
    except Exception as e:
        st.exception(e)


# ── Status row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Currents file", currents_path.name if currents_path.exists() else "—")
with col2:
    if currents_path.exists():
        meta = _load_currents_meta(str(currents_path))
        st.metric("Time steps", meta["n_times"])
    else:
        st.metric("Time steps", "—")
with col3:
    if currents_path.exists():
        st.metric("Grid", f"{meta['n_lon']}×{meta['n_lat']}")
    else:
        st.metric("Grid", "—")
with col4:
    latest = _latest_traj()
    st.metric("Latest run", latest.name if latest else "—")

if include_stokes:
    waves_label = waves_path.name if waves_path.exists() else "— (click 🌊 Fetch waves)"
    st.caption(
        f"🌊 Stokes drift active · scale={stokes_scale:.1f}× · waves file: `{waves_label}`"
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_drift, tab_accum, tab_currents = st.tabs(
    ["🌀 Drift", "🔥 Accumulation", "🌊 Currents"]
)


# ── Tab 1: Drift trajectories ────────────────────────────────────────────────
with tab_drift:
    ab_mode = st.session_state.get("ab_mode", False)
    ab_off = st.session_state.get("ab_off_path")
    ab_on = st.session_state.get("ab_on_path")
    ab_active = ab_mode and ab_off and ab_on and Path(ab_off).exists() and Path(ab_on).exists()

    if ab_active:
        df_off = traj_to_dataframe(load_trajectories(Path(ab_off)))
        df_on = traj_to_dataframe(load_trajectories(Path(ab_on)))

        mean_km, med_km, n = _stokes_shift_km(Path(ab_off), Path(ab_on))
        m1, m2, m3 = st.columns(3)
        m1.metric("Particles compared", n)
        m2.metric("Mean Stokes shift", f"{mean_km:.2f} km")
        m3.metric("Median Stokes shift", f"{med_km:.2f} km")

        st.markdown("##### A/B overlay")
        st.caption(
            "🔵 currents only  ·  🟠 currents + Stokes  ·  🟢 release point. "
            "Static view (no scrubber) — switch to ▶️ Run simulation for the time slider."
        )

        def _ab_path_layer(df, color):
            paths = (
                df.sort_values("time")
                .groupby("traj")
                .agg({"lon": list, "lat": list})
                .reset_index()
            )
            paths["path"] = paths.apply(lambda r: list(zip(r["lon"], r["lat"])), axis=1)
            return pdk.Layer(
                "PathLayer", data=paths, get_path="path",
                get_color=color, width_min_pixels=1.2, pickable=False,
            )

        def _ab_head_layer(df, color):
            head_df = df.sort_values("time").groupby("traj").tail(1)[["lon", "lat"]]
            return pdk.Layer(
                "ScatterplotLayer", data=head_df, get_position=["lon", "lat"],
                get_fill_color=color, get_radius=350,
                radius_min_pixels=2, radius_max_pixels=4,
            )

        seed_df = df_off.sort_values("time").groupby("traj").head(1)[["lon", "lat"]]
        seed_layer = pdk.Layer(
            "ScatterplotLayer", data=seed_df, get_position=["lon", "lat"],
            get_fill_color=[16, 185, 129, 220], get_radius=300,
            radius_min_pixels=2, radius_max_pixels=3,
        )

        deck = pdk.Deck(
            layers=[
                _ab_path_layer(df_off, [59, 130, 246, 130]),    # blue
                _ab_path_layer(df_on, [251, 146, 60, 180]),     # orange
                seed_layer,
                _ab_head_layer(df_off, [37, 99, 235, 230]),     # darker blue
                _ab_head_layer(df_on, [234, 88, 12, 240]),      # darker orange
            ],
            initial_view_state=_bbox_view_state(*bbox),
            map_provider="carto",
            map_style=map_theme.lower(),
            tooltip=False,
        )
        st.pydeck_chart(deck, width="stretch", height=600)

        if st.button("← Exit A/B view", help="Show only the latest single run"):
            for k in ("ab_mode", "ab_off_path", "ab_on_path"):
                st.session_state.pop(k, None)
            st.rerun()

    else:
        traj_str = st.session_state.get("traj_path")
        traj_path = Path(traj_str) if traj_str else _latest_traj()

        if traj_path is None or not Path(traj_path).exists():
            st.info("No simulation yet. Set parameters in the sidebar and click **▶️ Run simulation**.")
        else:
        ds = load_trajectories(traj_path)
        df = traj_to_dataframe(ds)
        n_traj = df["traj"].nunique()
        n_obs = df["obs"].nunique() if "obs" in df.columns else len(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Active trajectories", n_traj)
        m2.metric("Time steps", n_obs)
        if n_traj > 0:
            starts = df.sort_values("time").groupby("traj").head(1)
            ends = df.sort_values("time").groupby("traj").tail(1)
            # Mean great-circle-ish distance (small-angle approx)
            dlon = (ends["lon"].values - starts["lon"].values) * np.cos(
                np.radians(starts["lat"].values)
            )
            dlat = ends["lat"].values - starts["lat"].values
            mean_disp_km = float(np.mean(np.sqrt(dlon**2 + dlat**2))) * 111.0
            m3.metric("Mean displacement", f"{mean_disp_km:.1f} km")

        st.markdown("##### Particle paths")
        st.caption("Green = release  ·  Red = current/final position")

        # Animation slider — show progressive trail up to time index t
        times = sorted(df["time"].unique())
        t_idx = st.slider(
            "Time",
            min_value=0,
            max_value=len(times) - 1,
            value=len(times) - 1,
            help="Drag to scrub through the simulation",
        )
        t_now = times[t_idx]
        st.caption(f"📅 {pd.Timestamp(t_now).strftime('%Y-%m-%d %H:%M UTC')}")

        # Build trail data up to t_now
        trail_df = df[df["time"] <= t_now]
        paths = (
            trail_df.sort_values("time")
            .groupby("traj")
            .agg({"lon": list, "lat": list})
            .reset_index()
        )
        paths["path"] = paths.apply(
            lambda r: list(zip(r["lon"], r["lat"])), axis=1
        )

        head_df = (
            trail_df.sort_values("time")
            .groupby("traj")
            .tail(1)[["lon", "lat"]]
        )
        seed_df = df.sort_values("time").groupby("traj").head(1)[["lon", "lat"]]

        # ── pydeck layers ────────────────────────────────────────────────────
        path_layer = pdk.Layer(
            "PathLayer",
            data=paths,
            get_path="path",
            get_color=[59, 130, 246, 140],
            width_min_pixels=1.2,
            pickable=False,
        )
        head_layer = pdk.Layer(
            "ScatterplotLayer",
            data=head_df,
            get_position=["lon", "lat"],
            get_fill_color=[239, 68, 68, 230],
            get_radius=350,
            radius_min_pixels=2,
            radius_max_pixels=4,
        )
        seed_layer = pdk.Layer(
            "ScatterplotLayer",
            data=seed_df,
            get_position=["lon", "lat"],
            get_fill_color=[16, 185, 129, 200],
            get_radius=300,
            radius_min_pixels=2,
            radius_max_pixels=3,
        )

        deck = pdk.Deck(
            layers=[path_layer, seed_layer, head_layer],
            initial_view_state=_bbox_view_state(*bbox),
            map_provider="carto",
            map_style=map_theme.lower(),
            tooltip=False,
        )
        st.pydeck_chart(deck, width="stretch", height=600)


# ── Tab 2: Accumulation heatmap ──────────────────────────────────────────────
with tab_accum:
    traj_path = Path(st.session_state.get("traj_path") or "") if st.session_state.get(
        "traj_path"
    ) else _latest_traj()
    if traj_path is None or not Path(traj_path).exists():
        st.info("Run a simulation to see accumulation.")
    else:
        df = traj_to_dataframe(load_trajectories(traj_path))

        c1, c2 = st.columns([1, 4])
        with c1:
            mode = st.radio("Count", ["Final positions", "All positions"], index=0)
            n_bins = st.slider("Grid resolution", 30, 200, 80)
        only_final = mode == "Final positions"

        H, xedges, yedges = accumulation_grid(df, bbox, n_bins=n_bins, only_final=only_final)

        # Convert grid to pydeck-friendly point cloud (one row per non-zero cell)
        ny, nx = H.shape
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        XX, YY = np.meshgrid(xc, yc)
        mask = H > 0
        heat_df = pd.DataFrame(
            {
                "lon": XX[mask].ravel(),
                "lat": YY[mask].ravel(),
                "weight": H[mask].ravel().astype(float),
            }
        )

        with c2:
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=heat_df,
                get_position=["lon", "lat"],
                get_weight="weight",
                radius_pixels=60,
                intensity=1.0,
                threshold=0.03,
                aggregation="SUM",
            )
            deck = pdk.Deck(
                layers=[heatmap_layer],
                initial_view_state=_bbox_view_state(*bbox),
                map_provider="carto",
                map_style=map_theme.lower(),
            )
            st.pydeck_chart(deck, width="stretch", height=600)

        st.caption(
            f"Bin: ~{(xedges[1]-xedges[0])*111:.1f} km × {(yedges[1]-yedges[0])*111:.1f} km   ·   "
            f"max count per cell: {int(H.max())}"
        )


# ── Tab 3: Currents (the field driving everything) ───────────────────────────
with tab_currents:
    if not currents_path.exists():
        st.info("Fetch currents to see the underlying velocity field.")
    else:
        ds = xr.open_dataset(currents_path)
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_name = "latitude" if "latitude" in ds.coords else "lat"

        ti = st.slider(
            "Time", 0, ds.sizes["time"] - 1, ds.sizes["time"] - 1, key="curr_ti"
        )
        u = ds.uo.isel(time=ti).squeeze()
        v = ds.vo.isel(time=ti).squeeze()
        speed = np.sqrt(u**2 + v**2)

        st.caption(f"📅 {str(ds.time.values[ti])[:16]}")

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=ds[lon_name].values,
                y=ds[lat_name].values,
                z=speed.values,
                colorscale="Viridis",
                colorbar=dict(title="m/s", thickness=12),
                zmin=0,
                zmax=float(np.nanpercentile(speed.values, 99)),
            )
        )
        # Subsample arrows so it's not too dense
        step = max(1, len(ds[lon_name]) // 25)
        XX, YY = np.meshgrid(ds[lon_name].values[::step], ds[lat_name].values[::step])
        UU = u.values[::step, ::step]
        VV = v.values[::step, ::step]

        # Plotly doesn't have a native quiver — use line segments
        scale = 0.4 / max(np.nanmax(np.sqrt(UU**2 + VV**2)), 1e-3)
        xs, ys = [], []
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                if not (np.isfinite(UU[i, j]) and np.isfinite(VV[i, j])):
                    continue
                xs += [XX[i, j], XX[i, j] + UU[i, j] * scale, None]
                ys += [YY[i, j], YY[i, j] + VV[i, j] * scale, None]
        arrow_color = (
            "rgba(15,23,42,0.55)" if map_theme == "Light"
            else "rgba(255,255,255,0.55)"
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=arrow_color, width=1),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.update_layout(
            template="plotly_white" if map_theme == "Light" else "plotly_dark",
            height=600,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig, width="stretch")


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "DriftScope v0.2  ·  CMEMS currents + ERA5 Stokes drift  ·  OceanParcels"
)

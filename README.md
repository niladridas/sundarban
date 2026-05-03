# DriftScope — v0

> Lagrangian particle tracker for surface drift in the Bay of Bengal / Sundarbans
> region. Pick a bounding box, seed virtual particles, watch them advect on
> CMEMS surface currents. End-to-end pipeline: data → simulation → dashboard.

This is **module 1** of a larger ocean-transport stack. It's deliberately the
*spine* every later module (waves, tides, SPM settling, GOFLOW velocities,
SAR fronts) plugs into. Build this, validate it, then bolt the rest on.

---

## What you get

```
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌────────────┐
│  CMEMS API   │──▶│  OceanParcels    │──▶│  Trajectories   │──▶│ Dashboard  │
│  (currents)  │   │  (advection)     │   │  (NetCDF/Zarr)  │   │ (Streamlit)│
└──────────────┘   └──────────────────┘   └─────────────────┘   └────────────┘
```

- **Animated trajectories** on a satellite basemap (deck.gl / pydeck)
- **Interactive seeding** — click the map, pick a bbox, set particle count
- **Accumulation heatmap** — where do particles end up after N days?
- **Region-agnostic** — defaults to Sundarbans, works anywhere

---

## Quickstart

```bash
# 1. Environment
conda env create -f environment.yml
conda activate driftscope

# 2. Credentials — sign up free at https://marine.copernicus.eu
cp .env.example .env
# edit .env with your CMEMS username/password

# 3. Fetch currents for your region (default: Sundarbans, last 30 days)
python -m driftscope.fetch

# 4. Run a simulation
python -m driftscope.simulate

# 5. Launch the dashboard
streamlit run app.py
```

---

## Project layout

```
sundarbans-drift/
├── environment.yml          # conda env (parcels, copernicusmarine, streamlit)
├── .env.example             # CMEMS creds template
├── app.py                   # Streamlit dashboard
├── driftscope/
│   ├── __init__.py
│   ├── config.py            # regions, defaults
│   ├── fetch.py             # CMEMS data download
│   ├── simulate.py          # OceanParcels driver
│   ├── viz.py               # plots + animations
│   └── kernels.py           # advection kernels (extensible: waves, settling…)
├── data/
│   ├── currents/            # CMEMS NetCDFs land here
│   └── trajectories/        # simulation outputs
└── notebooks/
    └── 01_quickstart.ipynb  # end-to-end walkthrough
```

---

## Roadmap (next modules)

| # | Module                  | What it adds                                     |
|---|-------------------------|--------------------------------------------------|
| 1 | **Particle tracker** ✅  | Advection on gridded currents (this repo)        |
| 2 | Stokes drift            | Surface wave-induced transport                   |
| 3 | Windage                 | Wind-driven drift for buoyant material           |
| 4 | Tides                   | TPXO/FES barotropic tide overlay                 |
| 5 | SPM settling            | Vertical settling velocity for sediment          |
| 6 | GOFLOW currents         | Submesoscale velocity from thermal IR            |
| 7 | NIR SPM tracer          | Sentinel-2/3 SPM concentration as initial cond.  |
| 8 | SAR fronts              | Sentinel-1 convergence/film detection            |

Each is ~1 kernel + 1 data fetcher. The architecture is built for it.

---

## Why this stack

- **OceanParcels** — purpose-built for this; not reinventing advection
- **copernicusmarine** — official CMEMS Python client, handles auth + subsetting
- **Streamlit + pydeck** — fastest path to a "software product vibe" without writing JS
- **xarray + zarr** — chunked I/O so this scales when you add years of data

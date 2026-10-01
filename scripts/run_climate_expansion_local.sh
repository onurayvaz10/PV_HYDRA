#!/usr/bin/env bash
# Climate expansion on the local GPU machine (ERA5, raster and GPU available). Every step is resumable.
# Order: raw data -> screen (final decisions) -> queues -> semi-synthetic (CPU, GPU) -> rates (CPU, GPU) -> summary.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY scripts/fetch_climate_expansion.py                 # PVDAQ records (data/raw/climate_expansion, not committed)
$PY scripts/screen_climate_expansion.py                # rules 1-5 with ERA5 + raster; selection.csv
$PY scripts/run_climate_expansion.py queue             # queues/expansion_rate_jobs.csv
$PY scripts/run_semisynthetic_climate.py weather       # ERA5 at 16 sites + raster classes (>= 10 required)
$PY scripts/run_semisynthetic_climate.py queue         # queues/semisynthetic_climate_jobs.csv
$PY scripts/run_semisynthetic_climate.py cpu           # RdTools + kappa-HYDRA-D
$PY scripts/run_semisynthetic_climate.py gpu           # 7 ML models (GPU)
$PY scripts/run_climate_expansion.py rd                # RdTools reference + injections
$PY scripts/run_climate_expansion.py kappa             # kappa-HYDRA-D
$PY scripts/run_climate_expansion.py ml                # 7 ML models (GPU)
$PY scripts/run_semisynthetic_climate.py summary
$PY scripts/summarize_climate_expansion.py            # tables E1-E4 by tier

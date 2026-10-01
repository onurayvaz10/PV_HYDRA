#!/usr/bin/env bash
# GPU part of the climate expansion: kappa-HYDRA-D and the five neural performance models
# (lstm, gru, tcn, patchtst, informer). RdTools, XGBoost and random forest are run and committed from the cloud machine.
# Needs: git pull of the expansion branch (selection.csv, semisynthetic_sites.csv, configs/perf_models_hpo.json),
# internet (OEDI S3, Open-Meteo archive, Zenodo when Utrecht is selected), a CUDA build of PyTorch.
# Every step is resumable: rerun the script after an interruption. Expected run time on an RTX 4080: ~11 h (networks ~10 h, kappa-HYDRA-D ~1 h).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
NETS=lstm,gru,tcn,patchtst,informer
export RUN_TAG=_gpu                                  # own result files, no clash with the CPU workers
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA device')"
$PY scripts/fetch_climate_expansion.py --selected    # raw records of the selected systems (not committed)
$PY scripts/run_semisynthetic_climate.py gpu khd_full ${NETS//,/ }
$PY scripts/run_climate_expansion.py kappa
$PY scripts/run_climate_expansion.py ml --models=$NETS
echo "done. Commit only the result tables:"
echo "  git add results/climate_expansion/semisynthetic_climate_runs_gpu.csv results/climate_expansion/rates_kappa_gpu.csv results/climate_expansion/rates_ml_gpu.csv"
echo "  git commit -m 'Climate expansion: neural models (GPU)' && git push"

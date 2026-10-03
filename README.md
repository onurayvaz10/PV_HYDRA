# PV_HYDRA — machine-learning normalization and photovoltaic performance loss rates

Code and result tables for the study

> O. Ayvaz and Ö. Tomak, “Normalization-Model Dependence of Photovoltaic Performance Loss Rates: Benchmarking
> Machine-Learning Estimators Against Known Rates.”

Authors: **Onur Ayvaz** (ORCID 0009-0000-9522-7790) and **Özgür Tomak** (ORCID 0000-0003-2993-6913, corresponding
author), Department of Electricity and Energy, Technical Sciences Vocational School, Giresun University, Türkiye.

The repository contains:
- the estimators: seven machine-learning performance models with a year-on-year trend, the RdTools reference, PVUSA,
  STL and change-point baselines, and κ-HYDRA-D (PVWatts response × bounded time-blind correction × linear trend);
- the experiments: equal-budget tuning; injected trends on 18 measured systems at five locations with one fixed
  filter mask per system and their exact expected change; a 2 × 2 comparison of normalization model and trend
  estimator; reference variants (typical, listed and fitted temperature coefficient, clear-sky workflow);
  information-equal variants; exact-rate records on the weather of 20 countries from three generators, one of them a
  single-diode model independent of the estimator structure; assumption-violating records; ablations; loss profiles
  on the measured systems; and the earlier climate and geographic expansion;
- every result table (`results/`), the screening plans and the revision protocols with their dated amendments
  (`docs/`, `docs/protocols/`), and the scripts that regenerate the paper's figures and its number ledger (`paper/`).

No raw measurement data are included (see `DATA_NOTICE.md` and `LICENSES.md`).

## Environment

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; .venv/bin/activate on Linux/macOS
pip install -r requirements-degradation.txt
```

Experiments were run with Python 3.13 and PyTorch 2.11 on one NVIDIA RTX 4080 Laptop GPU (CUDA 12.8; LSTM and GRU
without cuDNN kernels).

## Quick check (minutes, no raw data)

| Command | What it does |
|---|---|
| `python paper/scripts/build_result_ledger.py` | recomputes every number printed in the paper from `results/` → `paper/result_ledger.csv` |
| `python paper/scripts/make_manifests.py` | system and split manifests → `paper/system_manifest.csv`, `paper/split_manifest.csv` |
| `python paper/scripts/make_paper_figures.py` | Figures 1–3 → `paper/figures/` |
| `python paper/scripts/test_eq2_truth.py` | noise-free test of the exact expected change of an injected trend, Eq. (2) |
| `python -m unittest tests.test_ml_plr -v` | unit tests of the reference period, outage screen, injection and known-rate recovery |
| `python scripts/verify_reproduction.py --quick` | regenerates the summary tables from the shipped run files and checks the number ledger |

## Full pipeline (about a day on one GPU)

| Step | Command | Output |
|---|---|---|
| 1. Data | `python scripts/fetch_dkasc.py`, `python scripts/fetch_tier_a.py`, `python scripts/fetch_climate_expansion.py` | `data/raw/` (not shipped) |
| 2. Sites and climate | `python scripts/make_site_table.py`, `python scripts/make_site_map.py` | `tables/T0*` |
| 3. Screen, drift rule, reference rates | `python scripts/run_rdtools_reference.py` | `tier_a_screen.csv`, `reference_plr.csv` |
| 4. Equal-budget tuning | `python scripts/run_ml_plr.py hpo` | `configs/perf_models_hpo.json` |
| 5. Two-stage rates and injections | `python scripts/run_ml_plr.py rd`, `python scripts/run_ml_plr.py run` | `ml_plr_runs*.csv` |
| 6. κ-HYDRA-D and ablations | `python scripts/run_ml_plr.py kappa` | `kappa_hydra_d_runs.csv` |
| 7. Semi-synthetic exact-rate records, intervals, coefficients | `run_semisynthetic.py`, `run_semisynthetic_ci.py`, `run_semisynthetic_gamma.py`, `summarize_semisynthetic_ci.py` | `T11`, `T16`, `T17` |
| 8. Assumption-violating records, baselines, equal-time budget, ablation | `run_semisynthetic_hard.py`, `diagnose_soiling_yoy.py`, `run_measured_baselines.py`, `run_equal_time.py`, `run_semisynthetic_ablation.py`, `summarize_robustness.py` | `T18`–`T21`, `T25` |
| 9. Forward and held-out-location tests (post hoc) | `run_independent_validation.py all`, `run_independent_transfer.py`, `run_gamma_sensitivity.py`, `summarize_independent_validation.py`, `summarize_forward_validation.py` | `T22`–`T24` |
| 10. Exact expected change, centring, site dependence, identifiability | `check_injection_truth.py`, `check_centering.py`, `robustness_checks.py`, `run_identifiability.py` | `T26`–`T31` |
| 11. Diagnostics | `run_refwindow_sensitivity.py`, `run_loss_structure.py`, `test_loss_structure.py`, `run_bin_diagnostic.py`, `run_transfer.py`, `run_shap.py`, `benchmark_cost.py` | `T7`, `T9`, `T10`, `T12`–`T15` |
| 12. Paper tables | `python scripts/summarize_degradation.py` | `T1`–`T11` |
| 13. Climate and geographic expansion | `screen_climate_expansion.py`, `run_climate_expansion.py` (GPU: `run_climate_expansion_gpu.sh`/`.ps1`), `run_semisynthetic_climate.py`, `summarize_climate_expansion.py`, `summarize_global.py`, `test_global_ranks.py`, `make_world_map.py` | `results/climate_expansion/tables/E*` |
| 13b. Fair reference and a non-PVWatts (ADR) response on the 20 weather sites | `run_climate_fair_reference.py a`, `run_climate_fair_reference.py b`, `run_climate_fair_reference.py summary` | `results/climate_expansion/tables/E12_fair_reference.csv` |
| 14. Fixed-mask benchmark (protocol P1–P3) | `run_fixed_mask.py rd`, `run_fixed_mask.py ml`, `run_fixed_mask.py kappa`, `run_fixed_mask.py kappa24`, `run_fixed_mask.py mlfull`, `run_refwindow_fixed.py` | `results/fixed_mask/*_runs*.csv` |
| 15. Reference variants (P4) | `run_reference_variants.py`, `run_reference_variants.py inject` | `results/fixed_mask/reference_variants*.csv` |
| 16. Single-diode generator (P5) and information-equal exact-rate runs (P3) | `run_singlediode_generator.py run`, `run_climate_info_equal.py` | `results/singlediode/`, `results/climate_expansion/info_equal_runs*.csv` |
| 17. Loss profiles on measured systems (P6) | `run_measured_profile.py` | `results/fixed_mask/profile_*.csv` |
| 18. Revision tables (P7) | `analyze_revision.py halves`, `analyze_revision.py`, `latitude_spearman.py` | `results/fixed_mask/tables/R*.csv` |
| 19. Run manifest | `python scripts/make_run_manifest.py` | `results/RUN_MANIFEST.csv` |

Each run script appends one row per finished unit, so an interrupted run resumes where it stopped.

## Design decisions

`docs/DEGRADATION_PAPER_PLAN.md` and `docs/CLIMATE_EXPANSION_PLAN.md` record the screening rules and analysis plans,
and every later amendment with its date and reason; `docs/protocols/` holds the revision protocols, each committed to
this repository before the corresponding runs (the commit time is the record). Analyses defined after the
retrospective results were known are labelled post hoc.

Fig. 1(a) uses the 1-km Köppen–Geiger map for 1991–2020 of Beck et al. (2023, CC BY 4.0), included in
`data/reference/climate/` with its legend and source record.

## Citation and licence

See `CITATION.cff`. The code is released under the MIT licence (`LICENSE`); data licences are listed in `LICENSES.md`.

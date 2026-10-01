# Degradation study — result digest (generated)

## Agreement with RdTools (inj. 0, across systems)

| method | mean |Δ| %/yr | Spearman ρ | n |
|---|---|---|---|
| gru | 0.180 | 0.99 | 15 |
| informer | 0.191 | 0.98 | 15 |
| lstm | 0.172 | 0.97 | 15 |
| patchtst | 0.184 | 0.98 | 15 |
| random_forest | 0.166 | 0.95 | 15 |
| tcn | 0.179 | 0.98 | 15 |
| xgboost | 0.164 | 0.97 | 15 |
| khd | 0.195 | 0.88 | 15 |

## Known-truth recovery |ΔPLR − expected| (%/yr), all rates

| method | MAE | bias |
|---|---|---|
| khd_no_correction | 0.019 | +0.010 |
| informer | 0.021 | -0.003 |
| rdtools | 0.022 | -0.007 |
| patchtst | 0.023 | -0.005 |
| lstm | 0.024 | -0.000 |
| khd_full | 0.027 | +0.021 |
| gru | 0.029 | +0.008 |
| tcn | 0.029 | -0.012 |
| xgboost | 0.030 | +0.004 |
| khd_no_physics | 0.036 | +0.026 |
| random_forest | 0.065 | +0.044 |
| khd_time_input | 0.260 | +0.126 |

Friedman (recovery error, 60 blocks, 9 methods): {'statistic': 56.066666666666606, 'p_value': 2.7377623502773357e-09, 'mean_rank': {'gru': 5.833333333333333, 'informer': 4.083333333333333, 'khd_full': 4.083333333333333, 'lstm': 4.8, 'patchtst': 4.45, 'random_forest': 6.766666666666667, 'rdtools': 4.116666666666666, 'tcn': 5.516666666666667, 'xgboost': 5.35}, 'nemenyi_cd': 1.5508651706516898, 'nemenyi_pairs': {'informer vs gru': 1.75, 'informer vs random_forest': 2.683, 'khd_full vs gru': 1.75, 'khd_full vs random_forest': 2.683, 'lstm vs random_forest': 1.967, 'patchtst vs random_forest': 2.317, 'rdtools vs gru': 1.717, 'rdtools vs random_forest': 2.65}}

## Performance-model fit and cost

| model | val_rmse | val_mae | val_r2 | parameters | train_s_shared_load | hpo_seconds |
|---|---|---|---|---|---|---|
| gru | 0.0268 ± 0.0119 | 0.0176 ± 0.0076 | 0.988 ± 0.013 | 1.518e+05 | 69.25 | 1917 |
| informer | 0.0279 ± 0.0121 | 0.0185 ± 0.0075 | 0.987 ± 0.013 | 8.154e+04 | 14.88 | 373.5 |
| lstm | 0.0272 ± 0.0118 | 0.0179 ± 0.0074 | 0.988 ± 0.013 | 1.373e+04 | 75.7 | 1128 |
| patchtst | 0.0297 ± 0.0125 | 0.0200 ± 0.0077 | 0.986 ± 0.014 | 2.586e+04 | 21.46 | 650.9 |
| random_forest | 0.0257 ± 0.0124 | 0.0156 ± 0.0071 | 0.989 ± 0.013 | 3.419e+06 | 1.376 | 64.19 |
| tcn | 0.0261 ± 0.0119 | 0.0170 ± 0.0073 | 0.989 ± 0.013 | 1.983e+05 | 6.751 | 132.8 |
| xgboost | 0.0258 ± 0.0130 | 0.0158 ± 0.0075 | 0.988 ± 0.014 | 813 | 4.17 | 267.6 |

## Common-domain fit (same held-out hours for all models)

| model | rmse_mean | rmse_std | mae_mean | mae_std | r2_mean | r2_std |
|---|---|---|---|---|---|---|
| gru | 0.0289 | 0.0138 | 0.0188 | 0.0092 | 0.9751 | 0.0293 |
| informer | 0.0292 | 0.0141 | 0.0189 | 0.0091 | 0.9746 | 0.0298 |
| khd-full | 0.0331 | 0.0164 | 0.0193 | 0.0085 | 0.9677 | 0.0362 |
| khd-no_correction | 0.0487 | 0.0152 | 0.0314 | 0.0104 | 0.9381 | 0.0438 |
| khd-no_physics | 0.0329 | 0.0169 | 0.02 | 0.0091 | 0.9675 | 0.037 |
| khd-time_input | 0.0256 | 0.0111 | 0.0154 | 0.0075 | 0.9809 | 0.0227 |
| lstm | 0.0292 | 0.0136 | 0.0192 | 0.009 | 0.9748 | 0.0292 |
| patchtst | 0.0315 | 0.0147 | 0.0207 | 0.0096 | 0.971 | 0.0322 |
| random_forest | 0.0284 | 0.0145 | 0.0178 | 0.0088 | 0.9754 | 0.0299 |
| tcn | 0.0279 | 0.0138 | 0.0177 | 0.0087 | 0.9766 | 0.028 |
| xgboost | 0.0287 | 0.0151 | 0.018 | 0.0091 | 0.9747 | 0.0312 |

## DM-HLN (Holm) — systems where model is significantly better / worse than reference

| reference | model | sig_better | sig_worse |
|---|---|---|---|
| khd-full | gru | 2 | 0 |
| khd-full | informer | 2 | 0 |
| khd-full | khd-no_correction | 0 | 14 |
| khd-full | khd-no_physics | 0 | 4 |
| khd-full | khd-time_input | 4 | 0 |
| khd-full | lstm | 1 | 0 |
| khd-full | patchtst | 1 | 2 |
| khd-full | random_forest | 2 | 0 |
| khd-full | tcn | 2 | 0 |
| khd-full | xgboost | 2 | 0 |
| xgboost | gru | 0 | 0 |
| xgboost | informer | 0 | 0 |
| xgboost | khd-full | 0 | 2 |
| xgboost | khd-no_correction | 0 | 13 |
| xgboost | khd-no_physics | 0 | 3 |
| xgboost | khd-time_input | 1 | 0 |
| xgboost | lstm | 0 | 1 |
| xgboost | patchtst | 0 | 6 |
| xgboost | random_forest | 0 | 0 |
| xgboost | tcn | 0 | 0 |

## κ-HYDRA-D ablation (inj. 0, mean over systems)

| variant | plr | plr_seed_sd | val_rmse | coverage_80 |
|---|---|---|---|---|
| full | -0.6021 | 0.01668 | 0.05471 | 0.8026 |
| no_correction | -0.6025 | 0.02385 | 0.06851 | 0.729 |
| no_physics | -0.592 | 0.01566 | 0.05401 | 0.8268 |
| time_input | -0.7616 | 0.176 | 0.04153 | 0.8098 |

## Transfer for young plants (r = -1 %/yr recovery; agreement with RdTools)

| model | condition | val_rmse | mae_vs_rdtools | recovery_mae | n |
|---|---|---|---|---|---|
| tcn | finetune_6 | 0.04019 | 0.1992 | 0.03329 | 15 |
| tcn | scratch_24 | 0.0261 | 0.1791 | 0.03292 | 15 |
| tcn | scratch_6 | 0.04894 | 0.1997 | 0.02944 | 15 |
| tcn | zero_shot | 0.07742 | 0.1381 | 0.02362 | 15 |
| xgboost | finetune_6 | 0.03901 | 0.1405 | 0.02303 | 15 |
| xgboost | scratch_24 | 0.02583 | 0.164 | 0.03903 | 15 |
| xgboost | scratch_6 | 0.03371 | 0.199 | 0.02226 | 15 |
| xgboost | zero_shot | 0.06802 | 0.1332 | 0.04698 | 15 |

## Semi-synthetic known truth (real weather, exact r): error = PLR - r (%/yr)

| family | method | bias | mae | max_abs | n |
|---|---|---|---|---|---|
| pvwatts | khd_full | -0.002402 | 0.008244 | 0.02799 | 24 |
| pvwatts | rdtools | -0.004302 | 0.009291 | 0.02337 | 24 |
| pvwatts | tcn | 0.01338 | 0.02015 | 0.04757 | 24 |
| pvwatts | xgboost | 0.03909 | 0.03909 | 0.1055 | 24 |
| thin_film | khd_full | 0.002084 | 0.008214 | 0.01892 | 24 |
| thin_film | rdtools | 0.04446 | 0.07044 | 0.1263 | 24 |
| thin_film | tcn | 0.01236 | 0.0175 | 0.05029 | 24 |
| thin_film | xgboost | 0.0487 | 0.0487 | 0.12 | 24 |

## Bin diagnostic: pooled RdTools PLR outside its own POA/T-bin range on 1 of 15 systems

## Hot/arid vs other (Tier B, system performance loss, %/yr)

| subset | n_hot_arid | median_hot_arid | n_other | median_other | mann_whitney_p |
|---|---|---|---|---|---|
| all | 36 | -0.812 | 67 | -0.8221 | 0.8114 |
| screened | 11 | -0.9719 | 18 | -1.678 | 0.05056 |

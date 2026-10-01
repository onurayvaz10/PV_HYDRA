# Degradation-focused SCI paper — study design (v1, 2026-09-28)

**User decisions (2026-09-28)**

- The paper targets **PLR (%/yr) and PR**, with **NREL RdTools YoY** as the reference method.
- Only open data. **The Karapınar SCADA data are not used.** PVOutput is excluded.
- The power-forecasting infrastructure is reused: screening, leakage controls, models, statistics, and the physical κ envelope.

The user's publication guidelines are binding. They are tracked in §7.

## 1. Research questions

**RQ1.** How accurately do machine-learning performance models recover the performance-loss rate (PLR) compared with RdTools YoY?

- Models: XGBoost, RF, LSTM, GRU, TCN, PatchTST, Informer, and the proposed physics-informed model.
- Accuracy is tested on **known, injected degradation** and on real multi-climate systems.

**RQ2.** Can future performance ratio (PR) be forecast 12 months ahead with calibrated uncertainty (quantiles), better than persistence and linear-trend baselines?

**RQ3.** Do hot and arid systems (BWh, BSk, BSh) lose performance faster than temperate and cold ones? Which drivers (SHAP) explain PR loss?

**RQ4.** Does a model trained on some plants transfer to unseen plants and climates (leave-one-location-out)?

## 2. Data (≥ 4–5 years, as the guidelines require)

| Tier | Data | Climate(s) | Length | Irradiance / temperature |
|---|---|---|---|---|
| A | DKASC Alice Springs, 12 fixed research arrays of different technologies (6 downloaded, 6 to add) | BWh (hot, arid) | 2008–2025 (≤ 17 y) | Measured GHI, tilted POA, ambient T |
| A | PVDAQ Golden CO 1430 / 1432 / 1433 | Dfb | 2009–2018 (≈ 8–9 y) | Measured POA, module T, ambient T |
| A | PVDAQ 1423 Henderson NV | BWh | 2015–2023 (≈ 7 y) | Measured POA, ambient T |
| A | PVDAQ Prize 2107 Arbuckle CA | Csa | 2017–2024 (≈ 7 y) | Measured POA, ambient T |
| A | PVDAQ Prize 9068 Kersey CO (tracker) | BSk | 2017–2025 (≈ 7 y) | Measured POA |
| A | PVDAQ 1199–1204 (MD, NJ, DE, PA) | Cfa | 2010–2020 | To be confirmed by the screen |
| B | PVDAQ residential, 91 systems (daily energy) | 14 classes | 8–13 y | ERA5 (reanalysis) normalisation |
| Ref. | DuraMAT *PV Fleet Degradation Insights* (DOI 10.21948/1842958) | USA fleet | — | Published RdTools results; OSTI unreachable 2026-09-28, retry |

Every Tier-A system passes the pre-specified physical screen before use: scale, night power, time-stamp convention, clear-day shape, and outages.

**Sensor drift.** Each Tier-A system's POA sensor is compared year by year with ERA5, and with PVGIS/SARAH where it covers the site. A trend in the ratio is reported and flags the system.

## 3. Methods

### 3.1 Reference PLR: RdTools

Sensor workflow (Tier A):

1. `normalize_with_expected_power`, using PVWatts with measured POA and a SAPM cell temperature.
2. Filters, each with a stated reason:
   - POA 200–1200 W m⁻²
   - clipping filter
   - cell-temperature range
   - normalised-energy range
   - maintenance and outage days
3. Insolation-weighted daily aggregation.
4. `degradation_year_on_year` with bootstrap CI.
5. Soiling via stochastic rate and recovery (SRR) for DKASC's dusty climate. This separates soiling from degradation.

Tier B uses YoY on ERA5-normalised daily energy with a clear-sky day filter. It is reported as **"system performance loss"**, not module degradation.

### 3.2 Two-stage ML PLR (RQ1)

1. For each system, a performance model f(POA, T_cell, T_amb, hour, day-of-year; 24-h input window) is trained on the **reference period only**: the first 24 *data months* (calendar months with ≥ 150 healthy daylight hours), counted from the first day on which the on-site POA sensor covers ≥ 80 % of hours over the next 30 days.
   - Training targets exclude outage and maintenance hours: hourly ratio P/(P_rated·POA/1000) outside [0.01, 2], and whole days whose ratio is < 30 % of the centred 30-day median.
   - Every 5th ISO week inside the reference period is held out (blocked) for early stopping, model scoring and HPO. PLR never enters the HPO objective.
2. The model is applied forward. The performance index is PI = P_measured / f(x).
3. RdTools YoY on PI gives PLR_model.

Evaluation:

- (i) Model fit on held-out reference-period data: RMSE, MAE, R², Diebold–Mariano.
- (ii) PLR error against RdTools.
- (iii) PLR error against **injected ground truth**.

### 3.3 Proposed physics-informed model: κ-HYDRA-D

κ-HYDRA-D is a physics-informed, degradation-aware network:

  P̂_q(t) = P_PVWatts(POA, T_cell) × g_q(x_t) × (1 + r·(t − t_mid)),  q ∈ {0.1, 0.5, 0.9}

*As implemented (supersedes the draft bullets below where they differ):* D(t) is linear in centred time, so r is the rate relative to mid-record performance and is directly comparable with YoY. g_q = 1 + 0.5·tanh(·) is a bounded correction whose inputs contain **no time or age variable**, so it cannot represent a monotone trend. Identifiability is therefore obtained by input design rather than by a decorrelation penalty. The quantiles are non-crossing by construction. The loss is an insolation-weighted pinball loss on hours passing the RdTools filters and the outage screen. The CI of r combines a leave-one-year-out jackknife with the seed spread. Ablations:
- `no_physics`: target P/P_rated, with no PVWatts term.
- `no_correction`: g is constant.
- `time_input`: a diagnostic in which g also sees time. The seed SD of r rises from 0.01 to 0.14 %/yr on DKASC 7, which shows the identifiability problem.

Components:

- **Physical backbone:** PVWatts expected power, a known physics model.
- **κ̂:** the neural correction for effects that are not degradation, such as spectrum, soiling state and angle of incidence. It reuses the κ-HYDRA encoder.
- **r:** an explicit annual rate, estimated **jointly** with κ̂.
- **Physics-informed losses:**
  - smoothness and monotonicity of D (no recovery in module degradation);
  - a penalty that stops κ̂ from absorbing any long-term trend: the yearly mean of κ̂ is decorrelated from time.
- **Uncertainty:** quantile heads (P10/P50/P90) for PR forecasts, plus an ensemble over ≥ 5 seeds for r.

The model outputs PLR directly, with an interval. It is compared with RdTools and with the two-stage ML approach.

### 3.4 PR forecasting (RQ2)

- Target: daily temperature-corrected PR. It is forecast 1–365 days ahead from the history.
- Training is global across Tier A and B systems.
- Baselines:
  - seasonal naive
  - linear trend
  - XGBoost, RF, LSTM, GRU, TCN, PatchTST, Informer
- Proposed: κ-HYDRA-D with quantiles.
- Metrics: RMSE, MAE, R², pinball loss / CRPS, P10–P90 coverage, DM-HLN.

### 3.5 Transfer and climate (RQ3, RQ4)

- **Transfer:** leave-one-location-out, in two variants: zero-shot and fine-tuned with the first 6 months.
- **Climate groups:** hot/arid (BWh, BSk, BSh) vs temperate/cold. Rates are compared with bootstrap CIs and a Mann–Whitney test.
- **SHAP:** applied to PR-loss drivers (temperature, irradiance, humidity, climate descriptors) and to model features.

### 3.6 Synthetic degradation (ground truth)

- A known rate r ∈ {−0.25, −0.5, −1.0, −2.0} %/yr is multiplied into measured power as the factor (1 + r·t) over the analysed record.
- *Change from v1:* no DKASC array has an RdTools CI that contains 0. The criterion is therefore the **recovered change**, ΔPLR = PLR(injected) − PLR(original), which must equal r. It is evaluated on every system, because every method is refitted on the injected data (the performance model sees the injected reference period, as it would in practice).
- Result: |ΔPLR − r| and CI coverage for every method.

## 4. Fair comparison and robustness (guidelines)

- **Same HPO budget for every model:** 20 Optuna TPE trials on the same validation split.
- **≥ 5 seeds** for every neural model, reported as mean ± SD.
- Chronological splits only. Scaling on training data only. Performance models are fitted on the first 24 months only.
- A compute table: parameters, training time, inference time.

## 5. Planned computation (FAST SCI)

- **Data:** about 20 Tier-A systems at 15 min → hourly; 91 Tier-B systems at daily resolution.
- **Two-stage performance models:** 8 models × ~20 systems.
  - Tree models: seconds each.
  - Neural models: global per system group, 5 seeds → ~40 runs.
- **PR forecasting:** 9 models × 5 seeds on global data → ~45 runs.
- **HPO:** 20 trials × 9 models = 180 short trials, on a development subset.
- **Expected:** a few GPU-hours; no API quota.

## 6. Contributions (draft; no "first-ever" wording)

1. **A multi-climate PLR benchmark on open data.** Sensor-grade Tier-A systems (17-year DKASC included) plus 91 systems in 14 climate classes. RdTools is the reference, and injected degradation gives known ground truth.
2. **A physics-informed, degradation-aware neural model (κ-HYDRA-D).** It estimates PLR and its uncertainty jointly with the performance model, and it is compared fairly with 7 ML/DL models under equal HPO.
3. **12-month PR forecasting with quantile uncertainty,** plus cross-site transfer.
4. **A hot/arid vs temperate comparison with SHAP drivers,** linked to O&M planning and warranty assessment.

## 7. Guideline compliance tracker

| Guideline | Status |
|---|---|
| Open data, multiple climates | Done: data downloaded (PVDAQ, DKASC, residential) |
| PLR / PR target, RdTools YoY reference | Designed; RdTools 3.2.1 installed |
| XGBoost, RF, LSTM, GRU, TCN, PatchTST, Informer | XGBoost, LSTM, TCN, PatchTST exist; RF, GRU, Informer to add |
| Novelty (≥ 1) | PINN-type κ-HYDRA-D, transfer, quantile, SHAP, hot/arid |
| RMSE, MAE, R², DM, ablation, multi-site | Implemented in `src/evaluation/stats.py`; extend with pinball / CRPS |
| ≥ 4–5 years | Tier A 7–17 y; Tier B 8–13 y |
| Confounders separated | RdTools soiling (SRR), clipping filter; wording "performance loss" where not separable |
| Filters stated (< 200 W/m², cloudy, maintenance, temperature correction) | In the RdTools pipeline (§3.1) |
| Sensor drift vs satellite | ERA5 / PVGIS cross-check (§2) |
| Equal HPO for all, 5–10 seeds, compute table | §4 |
| GitHub code | Reproducibility package (DKASC raw data excluded by its terms) |
| Literature gap table, contributions, cover letter, reviewers | Writing phase |

## 8. Implementation log (decisions made while running; every one is pre-specified before any PLR comparison)

| Date | Decision | Reason |
|---|---|---|
| 2026-09-28 | Sensor-drift check made hour-matched: monthly ratio of POA sensor to ERA5 GHI, divided by the calendar-month mean, then a linear trend. | The first version compared daily sums and flagged every DKASC array at about −4 %/yr. That was an artefact of the thinned logging after 2023 (fewer hours per day). The hour-matched trend is −0.04 to −0.06 %/yr: no drift. |
| 2026-09-28 | ERA5 is interpolated onto local whole hours. | Alice Springs is at UTC+9:30, so ERA5 hours fall on :30 local. |
| 2026-09-28 | The analysis start is the first sustained POA coverage. | DKASC logs power from 2008, but POA only from April 2013. |
| 2026-09-28 | The reference period is 24 *data* months. | Outage-heavy early years left seasons unrepresented, with only 2,395 healthy hours in the first 24 calendar months. |
| 2026-09-28 | Performance-model training targets are outage-screened. The PI stage keeps exactly the RdTools filters. | Unscreened targets gave R² = 0.48. Screened targets give R² = 0.97–0.995. |
| 2026-09-28 | The clock convention of each PVDAQ record is detected: on-site POA vs ERA5 GHI under local standard, local DST and UTC. All frames are re-stamped to local standard time. | Conventions differ between PVDAQ systems (e.g. 1432/1433/2107 use local DST). |
| 2026-09-28 | **Drift rule v2:** a system is flagged only when **both** references agree: hour-matched ERA5 **and** the clear-sky envelope (monthly P90 of POA / Ineichen GHI_cs, as a calendar-month anomaly), each beyond 1 %/yr and in the same direction. The rule was set **before** any PLR of the affected systems was computed. | ERA5 alone flagged all three co-located Golden systems. The second reference confirms 1432 (−2.8 %/yr) and 1433 (−5.3 %/yr), which stay excluded. 1430 is borderline (−1.1 ERA5 / −0.86 clear-sky) and passes; a sensitivity analysis without 1430 is reported. |
| 2026-09-28 | Excluded: 1199 and 1203 (no POA channel); 1200–1202 (POA logging stops in 2011–2012); 1204 (< 4 valid years); 9068 (< 4 valid years; night power > 1 % of rating). | Pre-specified screen. |
| 2026-09-28 | **One rate convention for every method:** the rate is taken relative to the first-year level, as in RdTools first-year-normalised YoY. κ-HYDRA-D estimates ρ relative to mid-record performance and converts it with r = ρ / (1 + ρ(0.5 − t_mid)). | Without this, the methods would estimate differently defined rates. In the synthetic test, the mid-record convention alone gives about 4 % relative bias. |
| 2026-09-28 | **Truth for real-data injections:** ΔPLR_expected = r + 2·d·r·t_c/100, where d is the method's own baseline PLR and t_c is the time centre of its data. The naive error ΔPLR − r is also reported. | A multiplicative injection interacts with the system's own degradation. In first-year-normalised YoY, E(t) − E(t−1) = d + r + 2dr·t_c. RdTools "misses" r = −1 by about +0.1 %/yr on DKASC, and this second-order term explains most of that. The correction uses each method's own d, so it favours no method. |
| 2026-09-28 | **Diagnostic finding.** On the thin-film arrays, the two-stage ML PLR is steeper than RdTools: dkasc_7 (CdTe) ≈ −1.05 vs −0.73 and dkasc_8 (a-Si) ≈ −1.0 vs −0.85; the mono-Si dkasc_18 shows no gap. For the RdTools PVWatts PI, the YoY PLR **within** every POA bin and temperature bin is steeper than the pooled value: dkasc_7 gives −0.75 to −1.8 across POA bins vs −0.71 pooled, and −1.42 (cool) vs −0.65 (hot). The share of hot hours varies between years (0.32–0.79). With an empirical effective γ, RdTools moves towards the ML value (−0.84 and −1.11). The time features explain only 0.07 %/yr of the gap. | The likely mechanism is a composition bias: PVWatts' linear γ and its lack of spectral/low-light terms mis-describe thin-film response, and the error projects onto the trend as the weather mix changes. Injected differences cannot detect this, because the bias cancels in ΔPLR. |
| 2026-09-28 | **Semi-synthetic known-truth benchmark added** (`scripts/run_semisynthetic.py`): real weather from 4 sites (BWh ×2, Dfb, Csa); synthetic power with exact r ∈ {−0.5, −1}; two true-response families, `pvwatts` and `thin_film` (weaker γ, ±3 % seasonal spectral term, low-light loss); 3 seeds; methods RdTools (datasheet γ), XGBoost, TCN and κ-HYDRA-D. | This tests the composition-bias mechanism against exact truth. Expectation, stated before the results: RdTools unbiased for `pvwatts`, biased for `thin_film`. |
| 2026-09-28 | **Correction to the diagnostic above.** Across all 15 systems (T10), the bin PLRs are steeper than the pooled value on almost every system, mono-Si included. The pooled value lies outside its bin range on only 1/15. The bin analysis therefore does **not** isolate a thin-film composition effect; it more likely shows within-bin selection and noise. The claim is withdrawn from the diagnostic, and only the semi-synthetic benchmark (exact truth) is used to test the mechanism. | Avoid over-interpreting a weak diagnostic. |
| 2026-09-28 | **Semi-synthetic result (192 runs, exact truth).** Error = PLR − r in %/yr:<br>• `pvwatts` family: κ-HYDRA-D bias −0.002 / MAE 0.008; RdTools −0.004 / 0.009; TCN +0.013 / 0.020; XGBoost +0.039 / 0.039.<br>• `thin_film` family: κ-HYDRA-D +0.002 / 0.008; RdTools +0.044 / 0.070 (sign depends on the site, −0.05 to +0.09); TCN +0.012 / 0.017; XGBoost +0.049 / 0.049.<br>• κ-HYDRA-D 80 % coverage: 0.807 and 0.810. | The pre-stated expectation is confirmed: RdTools is unbiased only when the true response is PVWatts. κ-HYDRA-D is the most accurate under both families. The steeper two-stage ML PLR on the real thin-film arrays is **not** reproduced here: on real dkasc_7, κ-HYDRA-D agrees with RdTools. That gap is therefore reported as unresolved (a candidate cause is thin-film transient behaviour inside the ML reference period) and not attributed to RdTools. |
| 2026-09-28 | **Reference-window dependence of two-stage PLR.** XGBoost with the reference window shifted from data months 1–24 to 13–36 and 25–48: dkasc_7 (CdTe) −1.08 → −1.01 → −0.82 (RdTools −0.73); dkasc_8 (a-Si) −0.97 → −0.91 → −0.90 (RdTools −0.85); dkasc_18 (mono-Si) −0.38 → −0.28 → −0.26 (RdTools −0.23). Extended to all 15 systems as T12 (`scripts/run_refwindow_sensitivity.py`, 5 seeds). | This explains the real-data gap without blaming RdTools. When the system's response shape changes over time (non-uniform degradation), a performance model fixed on an early window carries that change into the trend. RdTools and κ-HYDRA-D need no reference window. Practical message: report two-stage PLR together with its reference-window sensitivity. |
| 2026-09-29 | **Where κ-HYDRA-D and RdTools disagree, degradation is non-linear.** dkasc_11: the annual PI is flat (~0.97) until 2020, then falls to 0.84 by 2025. YoY gives −1.42 and OLS on the daily PI −1.40, while κ-HYDRA-D gives −0.86. dkasc_13: YoY −0.69, OLS −0.32, κ-HYDRA-D −0.45, driven by a 2025 drop in a partial year. Linear-ish systems agree (dkasc_7: −0.71 / −0.78 / −0.75). A non-linearity index (second-half minus first-half slope of the annual PI) will be added for every system. | A single PLR is estimator-dependent when degradation accelerates. The likely mechanism for κ-HYDRA-D: residuals from a linear trend line up with the late years' weather, so the correction g absorbs part of the acceleration through the climate inputs. Reported as a limitation. Candidate extension: piecewise or change-point rate. |
| 2026-09-29 | **Mechanism of the κ-HYDRA-D vs RdTools gap (corrects the g-absorption guess above).** The `no_correction` variant gives the same −0.87 on dkasc_11, so g is not the cause. Hourly κ quantiles by year on dkasc_11: in 2024–2025 the 10th percentile collapses to 0.47–0.58 while the median stays at 0.88–0.85. The losses concentrate in low-sun hours (before 09:00: 0.73; after 15:00: 0.83; midday: 0.85–0.87). This points to shading or a partial fault (bypass diode, string mismatch), not homogeneous module degradation. RdTools, as an insolation-weighted mean, includes these losses (**performance loss**). κ-HYDRA-D, as a median/pinball fit, is robust to intermittent partial-hour losses (**closer to module degradation**). The difference between the two is reported per system as an indicator of non-degradation partial losses. | This follows the guideline's rule on separating confounders (shading, faults): where they cannot be separated, the result is named "performance loss". |
| 2026-09-29 | **Pre-specified test (stated before T13 and the full κ-HYDRA-D results were seen).** Across the 15 systems, the signed difference κ-HYDRA-D − RdTools correlates positively with the tail index of T13: losses concentrated in a subset of hours make RdTools steeper than the median fit. Test: one-sided Spearman ρ > 0 at α = 0.05. The correlation with the non-linearity index is reported as secondary. | This checks the dkasc_11 mechanism across systems instead of relying on one anecdote. |
| 2026-09-29 | **Pre-specified test NOT supported** (T14, n = 15). Spearman of (κ-HYDRA-D − RdTools) with the tail index: ρ = 0.12, one-sided p = 0.33. With the non-linearity index (secondary): ρ = 0.26, p = 0.18. **Exploratory, labelled as such:** on the DKASC arrays, κ-HYDRA-D is less steep than RdTools in 9 of 12 (median +0.13 %/yr; Wilcoxon p = 0.034). Adding κ-HYDRA-D's outage-day screen to RdTools does not close the gap (median |gap| 0.133 → 0.139; p = 0.41; T15). | The dkasc_11 tail-loss mechanism does not generalise. The gap is reported as unresolved **estimator dependence**: a linear median fit on hourly data vs the median of year-on-year changes on insolation-weighted daily data. Both were accurate on the semi-synthetic records with linear degradation. The Discussion text on "performance loss vs module degradation" is revised accordingly. |
| 2026-09-29 | **Rate-interval coverage test added** (`scripts/run_semisynthetic_ci.py`): κ-HYDRA-D with five model seeds and the leave-one-year-out jackknife on the same 48 semi-synthetic records; coverage = share of records whose 95 % interval contains the exact rate. Compared with the bootstrap intervals of the year-on-year estimators (`scripts/summarize_semisynthetic_ci.py`, T16). | The hourly quantile coverage (0.81) says nothing about the rate interval. The claim "calibrated intervals" is kept only as far as this test supports it. |
| 2026-09-29 | **Nemenyi post-hoc test** added to the Friedman test of the recovery errors (critical difference of mean ranks at α = 0.05, 9 methods, 60 blocks: 1.55). Only TCN and random forest are separated from the four best-ranked methods (κ-HYDRA-D, LSTM, RdTools, Informer). | "κ-HYDRA-D ranked best" must not be read as "significantly better than the reference". |
| 2026-09-29 | **Target journal IEEE J. Photovoltaics; its Information for Authors applied:** abstract cut to 196 words (limit 200); complete author lists in every reference (no "et al."), with first and last pages, month and year (missing Crossref fields completed by hand in `reports/references_overrides.json`, each with its source); κ-HYDRA-D ablation and SHAP figures moved to the Supplementary Material (Figs. S1–S2) and the screening table folded into the text to approach the 6-page standard; graphical abstract (660 × 295 px) generated from the result files; builder support for the required author biographies and photographs. | JPV: standard length 6 pages (longer considered, slower); ORCID, biographies and photographs required. |
| 2026-09-29 | **Temperature-coefficient sensitivity of the reference** (`scripts/run_semisynthetic_gamma.py`, T17): RdTools re-run on the 48 semi-synthetic records with the true coefficient (oracle) and with a coefficient fitted to the first 12 months (P/(P_r·POA/1000) on T_c − 25, POA 600–1100 W/m²). Thin-film-like response, MAE (%/yr): datasheet 0.070, true 0.010, fitted 0.030 (fitted coefficient median −0.07 %/°C vs true −0.20: the fit absorbs the warm-season spectral gain). PVWatts response: all ≈ 0.009. | Pre-empts the objection that the datasheet coefficient is a straw man. The bias is dominated by the coefficient; a first-year fit only partly removes it; κ-HYDRA-D (0.008) matches the oracle without knowing the coefficient. |
| 2026-09-29 | **Rate-interval coverage result (T16, 48 records, complete).** κ-HYDRA-D 95 % intervals (five seeds + leave-one-year-out jackknife) covered the exact rate in 48 of 48 records (PVWatts 24/24, thin-film-like 24/24; Wilson 95 % interval per family 0.86–1.00); median width 0.090 / 0.092 %/yr; five-seed MAE 0.0054 / 0.0059 %/yr. Bootstrap intervals: RdTools 24/24 and 20/24, TCN 45/48, XGBoost 23/48. | The claim "calibrated intervals" is replaced by the measured coverage: the rate interval is reliable and on the conservative side (48/48 at nominal 95 %; P = 0.95^48 = 0.085, so over-coverage is not significant), wider than the bootstrap interval of the reference on the PVWatts records (0.09 vs 0.05 %/yr). Bootstrap intervals miss when the normalization is biased, because the bias is not part of the interval. |
| 2026-09-29 | **Fit vs rate accuracy quantified** (check_claims.py recomputes it): across the seven two-stage models, Spearman of common-domain RMSE (T3b) against known-truth recovery MAE (T2) = −0.71, p = 0.07; within single systems the correlation is positive on 4 of 15 (with κ-HYDRA-D added as an eighth model: −0.67 and 2 of 15, not used). | Supports the abstract's "fit accuracy did not predict rate accuracy" with a number instead of two anecdotes (random forest, time-input variant). Reported as "not positive", not as "better fit, worse rate" (p = 0.07). |
| 2026-09-29 | **Site identities and coordinates in the paper** (`scripts/make_site_table.py`, T0; main-text Table II, Table S1). Names, locations, coordinates, elevation, capacity and mounting come from the NREL PVDAQ catalogue and the DKASC listing; the DKASC coordinate (−23.7618, 133.8748, 546 m) is one approximate site point shared by the 12 arrays; climate classes from the 1-km 1991–2020 raster at each coordinate. | Requested by the authors; a reader must be able to locate every system. Cross-checked against an independent identity table in `reports/site_identity/` (identical coordinates). |
| 2026-09-29 | **Citation corrected: climate map.** The raster used is Beck et al. 2023 (1-km, 1991–2020; Sci. Data 10, 724; doi 10.1038/s41597-023-02549-6), not Beck et al. 2018; the reference was replaced. Golden is Dfb on this map, BSk in the PVDAQ catalogue; both are stated. | `data/reference/climate/source.json` records the 2023 dataset. |
| 2026-09-29 | **Label corrected: PVDAQ 1430 is the NREL Mesa single-axis tracker (720.7 kW), not a roof array.** `tier_a.py` labels 1430–1433 now follow the NREL catalogue and the tracking flag is read from it. The flag was not used by any computation (drift anomalies absorb geometry; measured POA; one SAPM mounting for all), so no result changes. | Metadata correctness. |
| 2026-09-29 | **Template leftovers removed** (`build_ieee_docx.py`): the header instruction line of the IEEE template is emptied (page number kept); the first-page footnote is now the template's real unnumbered footnote (support, corresponding author, affiliations) and the template's instruction footnotes are deleted; ORCID iDs are left to the submission system; the columns on the last page are balanced with a continuous section break. | IEEE template requirements; the instruction text must not reach the editor. |
| 2026-09-29 | **Reviewer-driven additions.** (1) Records that violate the estimators' assumptions (accelerating loss, 3 % step, unobserved soiling, ADR response; 336 runs): κ-HYDRA-D best with the ADR response (0.006) and among the best for accelerating loss, but no method handles the step or soiling; YoY median amplifies a soiling cycle (+0.22 to +0.86 %/yr on soiling-only records). (2) Non-ML baselines: PVUSA two-stage recovers injected trends like the ML models (0.036 %/yr); STL and change-point trend steps. (3) Mixed-effects model of recovery errors: only random forest differs from the reference. (4) κ-HYDRA-D with a mean loss reduces the DKASC gap only from 0.13 to 0.09 %/yr. (5) Equal-time tuning budget (374 s). (6) Claims narrowed to a hot desert climate; fit-vs-rate correlation reported as an observation. | External review: circularity of the semi-synthetic test, climate diversity, statistical power, budget definition, missing baselines, unexplained DKASC gap. |
| 2026-09-30 | **Forward and held-out-location validation merged (post hoc, same 15 arrays).** Runs by `run_independent_validation.py` / `run_independent_transfer.py` (600 + 600 runs), recomputed independently by `summarize_forward_validation.py` with equal location weighting (T22–T24). Later years: RMSE of the seven models 0.028–0.031 → 0.059–0.065; κ-HYDRA-D 0.073 and its hourly 80 % bands cover 43 % (bands are calibrated in the fitted period, stated as a limitation). Held-out location: zero-shot 0.094–0.106 vs 36 local months 0.061–0.063; fine-tuning 0.085–0.092; local XGBoost from 6 months 0.066. Test-period rate difference to the reference 0.13–0.18 (local) vs 0.19–0.33 %/yr (transferred). Reference field rates with γ × 0.8/1.2 change by up to 0.15 %/yr (median 0.04). Reported as post hoc, not as blind external validation; one manuscript rather than a second paper, because the result tempers rather than replaces the retrospective findings. PVOutput client in a separate working copy is not used and not released. | Tests generalization beyond the reference period and to unseen locations, which the retrospective design could not show. Text: abstract, Sections III-G, IV-I, V-E, VI; Tables VI, S22–S24. |
| 2026-09-30 | **JPV style, sites and κ-HYDRA-D ablation (reviewer items 1–4).** (a) Layout aligned with a JPV regular paper (vol. 16, no. 5, 2026; same template): numbered contributions, roadmap paragraph, full-width site table (Table II: country, latitude, longitude, elevation, 1-km Köppen–Geiger class, ERA5 GHI 2014–2023, on-site mean and daily-maximum air temperature) and a site map on the Köppen–Geiger raster (Fig. 1, `make_site_map.py`); on-site temperature used because the ERA5 cell at Arbuckle is about 3 °C warmer than the sensor. (b) κ-HYDRA-D name expanded (κ-ratio HYbrid Deep-Residual Anchoring for Degradation; proposed, authors to confirm) and a structure figure (Fig. 2, `make_khd_diagram.py`). (c) Ablation on exact-rate records (`run_semisynthetic_ablation.py`, 288 runs, T25, Table S21): without the correction network errors equal or exceed the reference (thin film 0.121, ADR 0.038 %/yr); without the physics term the full-record errors are unchanged (0.007, 0.009; p ≥ 0.23) but 3-year thin-film records worsen (0.013 → 0.022, p = 0.04) and soiling (0.112 → 0.142, p < 0.001). Reported as such: accuracy comes mainly from the time-blind correction fitted jointly with the trend; the physics anchor is a smaller, data-scarcity benefit. (d) Repository: code submitted as supplementary material for review, public DOI on acceptance ([PENDING] reference removed). (e) Equal-time budget: five seeds for XGBoost, RF, TCN, PatchTST (Informer unchanged); LSTM/GRU (3 and 6 trials, ~130 s per fit) seed 0 only and described as effectively untuned. | Reviewer asked whether the physics term matters; an honest ablation where the truth is exact answers it, and the site table/map make the climate scope explicit. |

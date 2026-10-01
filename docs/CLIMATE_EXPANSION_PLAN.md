# Climate expansion — pre-specified plan (2026-09-30, before any rate on a new system)

Reason: the supervisor's review — the measured records cover 4 sites and 3 Köppen–Geiger classes (13 of 15 systems
BWh). Goal: at least 8 classes, with the evidence tiered by the quality of the normalization data.

## Tiers

| Tier | Normalization | Minimum record | Role in the paper |
|---|---|---|---|
| S (sensor) | on-site plane-of-array irradiance and temperature | 4 years of valid days (unchanged) | main evidence (current Tables II–VI) |
| R (reanalysis) | ERA5 (Open-Meteo archive) irradiance and temperature at the site, transposed to the array plane with the published tilt/azimuth; on-site GHI where it exists (Hong Kong) | 3 years of valid days (YoY needs at least 2) | climate generalization, reported separately |
| D (daily) | ERA5 daily insolation | 8 years (existing 91 PVDAQ daily systems) | exploratory, as now (Table S10) |
| Semi-synthetic | exact rate on measured or ERA5 weather | — | known-truth accuracy per climate class |

Tier R rates carry the uncertainty of reanalysis irradiance; they are never pooled with Tier S in one statistic
without a tier term.

## Candidates (fixed now)

- **Tier S, PVDAQ parquet archive** (on-site POA required, verified from the channel metadata before download):
  Golden CO 4, 10, 33, 50, 51, 1208, 1283, 1289, 1332; Presque Isle ME 1239; Las Vegas NV 34, 35, 1276, 1277, 1278;
  Henderson NV 1367, 1368, 1369. Prize: Social Circle GA 9069, Kersey CO 9068 (re-screened with the full record).
- **Tier R**: New Orleans LA 1246, 1255, 1272; Port Chester NY 1217, 1221, 1225; Hong Kong stations (Zenodo 10909062);
  Utrecht NL systems (Zenodo 10953360); Maui HI 2105 (on-site GHI only).
- Not used: PVOutput (user decision), PVOD (records of 3–11 months), Karapınar SCADA and every dataset of other
  projects.

## Screening rules (fixed now)

1. The physical rules of the current screen (`tier_a.screen`): power scale, night offset, POA saturation, clock
   convention detected against ERA5.
2. Minimum record as in the tier table.
3. Tier S: the two-reference irradiance-sensor drift rule (ERA5 and clear-sky envelope, both > 1 %/yr, same sign).
   Tier R with on-site GHI: the same rule on GHI.
4. At most 3 systems per location in the main statistics (locations are the independent units); extra systems of a
   location are kept for sensitivity only, chosen by record length before any rate is computed.
5. Climate class from the 1-km Köppen–Geiger raster for 1991–2020 at the coordinates, not from metadata.

No rule is changed after a rate of a new system has been computed; any later amendment is recorded with its reason.

## Operational specification (2026-09-30, cloud machine; recorded before any rate of a new system)

These details were left open above; they are fixed here, before any new rate is computed, and no screening
rule above is changed. Code: `src/degradation/expansion.py`, `scripts/fetch_climate_expansion.py`,
`scripts/screen_climate_expansion.py`.

- **Downloads.** PVDAQ parquet archive, whole archived span; channels by the Tier-A precedence
  (`pvdaq_native.choose_power_metric`); weather channels matched case-insensitively (1283 logs `POA_irradiance`).
  Day folders dated before 1990 are ignored as timestamp faults (system 50 has one dated 1822). Tier-S candidates
  without a POA channel in their metrics metadata (1208, 1332, 1368, 1369) fail rule 1 and are not downloaded.
- **Prize systems.** 9068 as before (sum of the 8 inverter AC channels, pyranometer pad 1). 9069 (Social Circle
  GA, 38.7 MW DC): revenue meter 1 total AC power, first class-A pyranometer POA (02a), weather station 01
  ambient temperature. If the midday median meter2/meter1 ratio lies outside 0.95–1.05 the two meters are not
  redundant and their sum is used. 2105 (Maui): meter AC output, on-site GHI, ambient temperature; orientation
  "varies", so its irradiance input is the on-site GHI (horizontal approximation, as for Hong Kong).
- **Tier-R irradiance.** ERA5 GHI, DNI and DHI (Open-Meteo archive, hourly, interval-start labels), transposed to
  the catalogue tilt/azimuth with pvlib Perez 1990 (all-sites composite), albedo 0.2, solar position at the hour
  midpoint. A Tier-R system without a published tilt and azimuth fails (no transposition possible).
- **Tier-R temperature.** ERA5 2-m temperature (the only on-site channel, 1217 `module_temp_2`, is not used,
  consistent with the Tier-A loader's channel set).
- **Tier-R rule 1.** Scale p99.9(P) ≤ 1.10 × DC; night p95(|P|) ≤ 1 % DC over hours with irradiance input
  < 5 W/m²; valid day = ≥ 8 hours with irradiance input > 50 W/m² and complete power (the Tier-S definition with
  the irradiance input in place of the sensor); clock convention = the Tier-A hypotheses (local standard, local
  DST, UTC) scored by the correlation of AC power (or on-site GHI) with the ERA5 reference.
- **Pending state.** When ERA5 or the Köppen raster cannot be reached, the affected checks are written as pending
  and the decision is "pending"; rules that do not depend on ERA5 (scale, Tier-S night/sensor/length) are final.
  A convention scored against the pvlib clear-sky model is kept only as a diagnostic, as are clear-sky
  estimates of the Tier-R night offset and valid years.
- **Location.** The catalogue `site_location` (normalised). Systems already in the paper's main set count toward
  the limit of three (Golden 1430, Henderson 1423). Las Vegas and Henderson are separate locations (25 km apart);
  a sensitivity analysis merges them.
- **Hong Kong.** Stations of Zenodo 10909062 present under `data/raw/external/hongkong/` are Tier-R candidates with
  the territory's on-site GHI as irradiance input (the dataset's shared 1-min weather); all stations form one
  location, "Hong Kong". Utrecht has no power loader yet; its format is to be checked on the local copy.
- **Summaries.** `scripts/summarize_climate_expansion.py`: main-role systems only; errors averaged within a location,
  then over locations; Tier S and Tier R in separate rows; the only pooled statistic is an OLS of the location-level
  absolute error on method and tier (cluster-robust by location).
- **Semi-synthetic climate set.** `configs/semisynthetic_climate_sites.csv`: 16 weather sites chosen for their
  expected class; the class used is the raster class; the run needs ≥ 10 distinct raster classes. ERA5 hourly
  2015-01-01 – 2022-12-31 at every site; POA transposed to latitude tilt facing the equator (Perez, albedo 0.2);
  the generator, rates (−0.5, −1.0 %/yr), families (pvwatts, thin_film) and seeds (0–2) of
  `scripts/run_semisynthetic.py`; truth = the injected rate.

### Amendments (dated, with reason; no rate of a new system had been computed)

- 2026-09-30 — 9069 meters. The two revenue meters never log at the same time (meter 1 2016-02 – 2019-04,
  meter 2 2020-03 – 2023-11; 0 overlapping samples), so the redundancy/sum rule above cannot be applied. Meters
  with < 1 % overlapping samples are treated as a meter replacement and spliced (meter 1, then meter 2); the change
  is recorded for a step check. The registered POA channel (class-A pyranometer 02a) ends in 2019 and has
  p99.9 = 1712 W/m², so 9069 fails rule 1 ("no usable on-site POA"). The channel is **not** changed after this
  was seen; choosing another POA channel (e.g. reference cell 01, 92–99 % coverage every year) would be a new,
  data-informed decision for the authors to make and record here.
- 2026-09-30 — catalogue time-zone codes (7, 5, 10 = hours west of UTC) are mapped to America/Denver,
  America/New_York and Pacific/Honolulu so that the same clock hypotheses apply as for named zones.
- 2026-09-30 — Utrecht (Zenodo 10953360), specified before any Utrecht data were screened: the unfiltered
  1-min power file (our own screen is applied, as for PVDAQ; the authors' filtered files are not used), hourly means of
  complete hours (≥ 90 % of the 1-min samples), estimated DC capacity and tilt/azimuth from `metadata.csv`, ERA5 at the
  nearest 0.25° cell over 2013-12-31 – 2018-01-01, ERA5 2-m temperature; all 175 systems are Tier-R candidates of one
  location, "Utrecht" (rule 4: three main systems by valid years). Hong Kong stations share one ERA5 cell the same way.
- 2026-09-30 — run scope (compute): main-role systems get the full protocol (RdTools, κ-HYDRA-D, 7 ML models × 5
  seeds × 5 injections); sensitivity systems of a location, ranks 4–6 by record length only, get RdTools and κ-HYDRA-D.

### Screen outcome (2026-09-30; rules as registered, no amendment)

ERA5 (Open-Meteo) and the 1-km Köppen–Geiger 1991–2020 raster were used for every decision below.

- Tier S: pass — Golden 4, 10, 33, 1283, 1289 (Dfb) and Las Vegas 1278 (BWh); fail — 14 (see `screen_tier_s.csv`).
- Tier R: pass — New Orleans 1246, 1255, 1272 (Cfa; 3.41–3.56 valid years) and Port Chester 1217, 1221, 1225
  (Dfa; 3.24). Fail — Maui 2105 (night power 9.2 % of rating; on-site GHI clock = UTC), all 60 Hong Kong stations
  (< 3 valid years: the dataset spans 2021–2023) and all 175 Utrecht systems.
- **Utrecht is excluded by the valid-day definition, not by data quality.** A valid day needs ≥ 8 hours with
  irradiance > 50 W/m². At 52° N the winter days are too short for that: even with complete data, four years of
  Utrecht give at most 2.9 valid years (clear-sky upper bound, `valid_years_clearsky_upper_diag`); the best system has
  2.63. The rule was registered before any Utrecht data were seen, but its outcome was seen before this note, so
  relaxing it (for example a threshold relative to day length) would be a data-informed amendment; it is
  left to the authors. No Utrecht rate was computed.
- Measured classes after expansion: BWh, Dfb, Csa (paper) + Cfa, Dfa (Tier R) = 5 of the targeted 8. The
  semi-synthetic test covers 13 raster classes (`semisynthetic_sites.csv`).
- Amendment of scope (compute): κ-HYDRA-D and the five neural networks run on the GPU machine
  (`scripts/run_climate_expansion_gpu.{sh,ps1}`); RdTools and the tree models run in the cloud machine. No rule change.

- 2026-09-30 — scope fallback without a GPU (compute only, no screening rule changed). The five neural performance
  models need a GPU for the full protocol. If it is not available, they run on CPU with `REDUCED=1`: seed 0 instead of
  seeds 0–4, injections 0 and −1 %/yr instead of five levels, and in the semi-synthetic test the rate −1 %/yr and seed
  0 only. κ-HYDRA-D, RdTools and the tree models always run the full protocol. Results of a reduced run are labelled
  as such in every table; a later full run (`scripts/run_climate_expansion_gpu.{sh,ps1}`) only adds the missing units.
  Not used: the authors chose the full protocol on their GPU (2026-09-30); no reduced-scope result enters the paper.

### Amendments of 2026-10-01 (data-informed where stated; no rate of a new system had been computed)

- **Tier D restored (crowd-sourced daily systems).** The 91 daily PVDAQ systems (ids >= 10000) carry public names
  "Pvoutput.org ..." : they are crowd-sourced records originally contributed through PVOutput.org and redistributed by
  NREL in the OEDI PVDAQ archive under CC BY 4.0 (DOI 10.25984/1846021). Only the NREL OEDI copy is used (no PVOutput
  API, no `data/raw/pvoutput` request ledger); the origin is stated in the paper. Decision by the authors, same day:
  use them because the licence permits it. Their rates are "system performance loss" on ERA5-normalised daily energy,
  exploratory, reported separately (never pooled with Tier S/R).
- **Tier R-short (data-informed).** The 3-year rule excluded all Hong Kong stations (records 2021-2023) and all
  Utrecht systems (clear-sky bound 2.9 valid years at 52 N) although YoY needs only two years. For these two external
  datasets the minimum is lowered to 2.0 valid years, everything else in rule 1-3 unchanged, and their systems are
  labelled R-short and summarised in separate rows. The outcome of the 3-year screen had been seen; this is stated in the paper.

- **Semi-synthetic site set replaced (2026-10-01, before any semi-synthetic run; the cloud's interim 16-site runs are
  archived in `revision/_cloud_interim_16sites_20261001`, not used).** Design rule set by the authors: at most one
  site per country, sites on all inhabited continents, at least 10 Köppen-Geiger classes. 20 sites in 20 countries
  (`configs/semisynthetic_climate_sites.csv`, column `country`); the class used is the raster class.
- **Country as the independent unit in the measured summaries.** The measured records come from few countries (United
  States, Australia, Netherlands, Finland, Hong Kong); the headline statistics weight countries equally (locations
  within a country are averaged first), with the location-weighted statistics as a sensitivity analysis.
- **Finland (FMI outdoor solar laboratories, CC BY 4.0, doi:10.57707/fmi-b2share.tn0wb-as670)** added as a Tier-S
  candidate source (on-site plane-of-array irradiance, module temperature); screened with the Tier-S rules.
- **Africa and South America:** no open multi-year measured plant records were found (Brazil BR-PVGen 14 months,
  Safi Morocco about one year, Colombia one year); these continents enter through the semi-synthetic tests only.

- **Finland screen outcome and R-short/S-short rule (2026-10-01).** Helsinki (15 deg POA pyranometer, 3.34 valid years),
  Kuopio (15 deg POA, 2.24) and Sodankyla 20 deg (GHI only, 2.26) failed the 4-year (S) and 3-year (R) rules. Same
  data-informed relaxation as for Hong Kong and Utrecht: minimum 2.0 valid years and 24 matched months for the
  sensor-drift rule (`tier_a.MIN_DRIFT_MONTHS`, default 36 unchanged for the paper's Tier A); labelled S-short (Helsinki,
  Kuopio) and R-short (Sodankyla), summarised in separate rows. Hours with a visual snow status >= 2 are removed.
  Nameplate capacity is not published in machine-readable form: DC capacity is estimated from the record (p99 of DC
  input at POA > 700 W/m2), which only fixes the normalisation. Azimuth assumed south (clear-sky diagnostic only).
  All three pass the screen. No Finnish rate had been computed.

- **Short-record ML protocol (2026-10-01, before any ML rate of a short record).** The seven performance models learn
  from a 24-month reference period. Hong Kong (31 months), Kuopio and Sodankyla leave no post-reference span with 24
  months. For the short-record systems (hk_, ut_, fmi_) the reference period is 12 months for all seven models;
  RdTools and kappa-HYDRA-D are unchanged. The four PVDAQ tiers keep 24 months. Short-record rows are reported separately.
- **Finnish night hours.** The FMI inverters log nothing at night; hours with on-site irradiance < 5 W/m2 and no power
  record are set to zero output (complete hourly grid for the 24-h input windows). Snow hours stay removed.

- **Hong Kong irradiance input (2026-10-01, data-informed: found when the first Hong Kong rates were inspected).** The
  territory's shared on-site GHI sensor jumps by +37 % in 2023-02 (six-month means before and after, relative to ERA5 GHI)
  while the inverter power is unchanged; every method then returned -22 to -25 %/yr on all three main stations (a sensor
  step, not degradation; the registered two-reference drift rule tests trends, not steps). The on-site GHI is not used for
  any Hong Kong station: the irradiance input is ERA5 GHI (horizontal approximation, as for any Tier-R system without a
  published orientation), temperature as before. All Hong Kong stations are screened again with this input (valid years,
  night offset, clock) and the main set is reselected by the registered ranking. Rates computed with the faulty sensor
  were discarded (kept in revision/ as *.before_hk_era5). A step check (largest six-month change of the on-site / ERA5
  ratio) was run for every other on-site sensor: Helsinki 4.5 %, Kuopio 7.8 %, Sodankyla 4.1 %, Golden 10 13.5 % and
  Golden 33 9.1 % within 6-23 years of record; none is a step of the Hong Kong kind.

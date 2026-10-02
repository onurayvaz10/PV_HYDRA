# Revision protocol — written and committed before the corresponding runs (2 October 2026)

Authors: Onur Ayvaz and Özgür Tomak. Every result of these analyses is reported whatever it shows. A change after
results are seen is recorded as a new dated version with its reason.

## P1. Injection experiment with a fixed filter mask
- **Reason.** In the earlier injection runs the filter mask was recomputed on the injected power; the quantile
  clipping filter and the outage screen depend on power, so each injected rate (and each method) saw a different set
  of hours. On the Arbuckle system the recovered change was 73–85 % of the expected change.
- **Design.** For each system the RdTools filter mask (reference normalization, technology-typical temperature
  coefficient) and the target hygiene are computed once on the unmodified record and applied to every injected rate
  and every method. Injected rates −0.25, −0.5, −1, −2 %/yr (and 0); reference, seven two-stage models (5 seeds),
  κ-HYDRA-D (5 seeds; jackknife at r = 0) and its no-physics and no-correction variants (r = 0 and −1).
- **Systems (P9).** 18 systems at 5 locations: the 15 long sensor-grade systems and the three long expansion systems
  with on-site POA (Golden 10 and 33, Las Vegas 1278), selected by the rule written before any rate (≥ 3 valid years,
  on-site POA). ERA5-normalized and short records move to the supplementary material.
- **Arbuckle criterion.** Recovery ratio (observed change / expected change of Eq. (2)) at the four injected rates
  in 0.95–1.05. If not met, the run is repeated with the clipping filter disabled for that system; if still not met,
  the system is reported separately with the reason (apparent rate +1.02 %/yr, interval above zero).

## P2. 2 × 2 comparison of normalization and trend estimator
Cells per system (unmodified record, fixed mask): A = PVWatts + YoY (reference); B = PVWatts + linear median trend
(κ-HYDRA-D without correction); C = learned normalization PI = P / (P_PVW · g_0.5) + YoY; D = κ-HYDRA-D.
Normalization effect = [(C − A) + (D − B)] / 2; estimator effect = [(B − A) + (D − C)] / 2; median absolute value and
maximum over systems.

## P3. Information-equal comparison
κ-HYDRA-D with g trained on the first 24 data months (ρ = 0), then frozen while ρ is fitted on the whole record;
XGBoost and TCN trained on the whole record (every fifth ISO week held out, no time input). Run on the injection
experiment (fixed mask, 5 seeds), on the 240 exact-rate records of the 20 weather sites and on the P5 records.

## P4. Reference variants
Reference YoY with the technology-typical coefficient, with the module datasheet coefficient where a manufacturer
datasheet for the installed module is found, and with a coefficient fitted to the first 12 months (regression of
P/(P_r·POA/1000) on T_c − 25 at POA 600–1100 W/m²). The RdTools clear-sky workflow is run where the array geometry
is documented or can be estimated from the record; otherwise this is stated.

## P5. Independent generator (single-diode)
pvlib CEC single-diode model (calcparams_cec + singlediode) for one c-Si and one CdTe module of the CEC database,
with incidence-angle and spectral corrections; degradation through parameters: series resistance increasing and
photocurrent decreasing linearly with time, scaled so that P_mp at standard conditions changes by −0.5 or −1 %/yr
(the truth is the annual change of P_mp at standard conditions); day-to-day AR(1) multiplicative noise (φ = 0.7),
1 % irradiance-sensor scale error, inverter clipping at DC/AC = 1.25. ERA5 weather 2015–2022 at the 20 sites; 2
modules × 2 rates × 3 seeds = 240 records. Methods: reference, seven two-stage models, κ-HYDRA-D (5 seeds +
jackknife), and the P3 information-equal variants. κ-HYDRA-D rate-interval coverage reported with a Wilson interval.

## P6. Identifiability on measured systems
For each of the 18 systems, ρ fixed at 9 values within ±1 %/yr of the free fit; g refitted at each value (seed 0);
training loss recorded. Profile width: range of ρ where the loss is within 5 % of its minimum. Linear trends of the
annual mean POA and ambient temperature reported per system.

## P7. Statistics
Practical-equivalence margin δ = 0.05 %/yr (about one sixth of the median 95 % interval width of a measured rate).
Measured results summarized per location; no between-location test with fewer than five independent units. The
mixed-effects model uses system nested in location as random effect.

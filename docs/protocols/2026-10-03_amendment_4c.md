# Amendment 4c (3 October 2026; the commit time is the record)

Recorded after the first run of item 1 of amendment 4 (interval calibration under other noise structures) and
before its rerun.

**Defect.** The noise generator of `scripts/run_calibration_stress.py` was seeded by noise mode and seed only, so all
20 weather sites received the same noise sequence. With day-to-day AR(1) noise this made the 40 records two noise
realizations: every record had an error of the same sign. The first run is kept unchanged in
`results/calibration_stress_v1_shared_seed/` and is not used for any statement.

**Rerun.** The noise is seeded per site, mode and seed (CRC-32 of "site|mode|seed"); design, methods and outcomes are
those of amendment 4, item 1. The summary also reports, per mode and method, the largest share of errors with one
sign as a check that the noise realizations are independent across sites.

**Related note.** The 1 % independent hourly noise of the main exact-rate records (`scripts/run_semisynthetic.py`,
also used by the generators of the 20 weather sites) is likewise seeded by seed only. Independent hourly noise of 1 %
contributes little to a rate error that is dominated by the weather and the response model; the rerun of the
independent-noise modes above quantifies the size of noise-driven rate errors with independent realizations.

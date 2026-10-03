# Amendment 4b (3 October 2026; the commit time is the record)

Recorded after the Tier-S screen of the US DOE RTC baseline systems (item 3 of amendment 4) and before any rate of
these systems was computed.

**Screen result.** No system passed: Williston VT 2.41 valid years (both inverters); Albuquerque NM 3.03 valid years
(both inverters; every other rule passed); Cocoa FL 3.27 and 3.29 valid years and night power above 1 % of the
rating; Henderson NV (excluded a priori) 2.35 valid years. The pre-specified extension set is therefore empty, and
this is reported as the result of item 3 (`results/fixed_mask_rtc/rtc_screen.csv`).

**Exploratory extension.** Systems that fail only the record-length rule of the Tier-S screen and have at least 3
valid years — the record-length rule under which Golden 10 and 33 and Las Vegas 1278 entered the main benchmark —
are run with the full design of item 3 and reported as an exploratory extension, labelled as such and never pooled
with the main benchmark. This admits the two Albuquerque inverters (one location, climate BSk); the rule is applied
by `scripts/screen_rtc.py` (column `exploratory`), not by hand.

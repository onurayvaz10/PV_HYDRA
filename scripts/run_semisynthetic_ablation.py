"""κ-HYDRA-D ablation on records with an exact rate: does the PVWatts (physics) term matter?

The ablations of Table S6 were run on measured records with injections, where the no-physics variant gave almost
the same rate as the full model. Injections cannot reveal a bias of the normalization model, so the same variants are
run here on the semi-synthetic records of run_semisynthetic.py (PVWatts and thin-film-like responses) and on the
assumption-violating records of run_semisynthetic_hard.py (knee, step, soiling, ADR response), with the identical
generators, seeds and κ-HYDRA-D hyper-parameters (one seed per record, as for khd_full there).
Variants: no_physics (target P/P_rated, g carries the whole response), no_correction (PVWatts term and trend only).
Output: results/degradation_paper/semisynthetic_ablation_runs.csv (resumable; the full model is read from the
existing runs, not refitted).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_semisynthetic as semi  # noqa: E402
import run_semisynthetic_hard as hard  # noqa: E402
from src.degradation.kappa_hydra_d import estimate  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

OUT = ROOT / "results/degradation_paper/semisynthetic_ablation_runs.csv"
VARIANTS = ["no_physics", "no_correction"]
COLUMNS = ["set", "weather", "climate", "case", "rate", "seed", "variant", "plr", "truth", "error", "val_rmse",
           "fit_seconds"]


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    keys = set(zip(done["set"], done.weather, done.case, done.rate.astype(float), done.seed, done.variant))
    for raw in (semi.WEATHER[::-1] if "--reverse" in sys.argv else semi.WEATHER):      # a second worker runs backwards
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        jobs = [("semi", fam, rate, seed) for fam in ("pvwatts", "thin_film") for rate in semi.RATES for seed in semi.SEEDS]
        jobs += [("hard", sc, float("nan"), seed) for sc in hard.SCENARIOS for seed in semi.SEEDS]
        for kind, case, rate, seed in jobs:
            key_rate = -99.0 if kind == "hard" else rate
            todo = [v for v in VARIANTS if (kind, wid, case, key_rate, seed, v) not in keys]
            if not todo:
                continue
            if kind == "semi":
                syn, truth = semi.synthesize(frame, meta, case, rate, seed), rate
            else:
                syn, truth = hard.synthesize(frame, meta, case, seed)
            rows = []
            for v in todo:
                r = estimate(syn, 5.0, semi.G_DATASHEET, seeds=(seed,), jackknife=False, variant=v, **hp)
                rows.append({"set": kind, "weather": wid, "climate": meta["climate"], "case": case, "rate": key_rate,
                             "seed": seed, "variant": v, "plr": r["plr"], "truth": truth, "error": r["plr"] - truth,
                             "val_rmse": r["val_rmse"], "fit_seconds": r["fit_seconds"]})
                print(wid, kind, case, rate, seed, v, round(r["plr"], 3), "truth", round(truth, 3),
                      f"{r['fit_seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


SHORT_YEARS = 3.0


def short_records() -> None:
    """Data-scarce records: the first 3 years of the thin-film-like (r = -1 %/yr) and ADR records. With little data
    the correction network must learn the response from few weather situations, which is where an anchor could help.
    The full model and the reference are run as well (same records), so the comparison is paired."""
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    from src.degradation.rdtools_pipeline import sensor_plr
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    keys = set(zip(done["set"], done.weather, done.case, done.seed, done.variant))
    for raw in (semi.WEATHER[::-1] if "--reverse" in sys.argv else semi.WEATHER):
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        for case in ("thin_film", "non_pvwatts"):
            for seed in semi.SEEDS:
                todo = [v for v in ["full", *VARIANTS, "rdtools"] if ("short", wid, case, seed, v) not in keys]
                if not todo:
                    continue
                if case == "thin_film":
                    syn = semi.synthesize(frame, meta, case, -1.0, seed)
                else:
                    syn, _ = hard.synthesize(frame, meta, case, seed)
                syn = syn[syn.index < syn.index.min() + pd.Timedelta(days=365.25 * SHORT_YEARS)]
                truth = -1.0                                        # linear trend in both generators
                rows = []
                for v in todo:
                    if v == "rdtools":
                        r = sensor_plr(syn, 5.0, semi.G_DATASHEET)
                        r.update(val_rmse=float("nan"), fit_seconds=float("nan"))
                    else:
                        r = estimate(syn, 5.0, semi.G_DATASHEET, seeds=(seed,), jackknife=False, variant=v, **hp)
                    rows.append({"set": "short", "weather": wid, "climate": meta["climate"], "case": case, "rate": -1.0,
                                 "seed": seed, "variant": v, "plr": r["plr"], "truth": truth, "error": r["plr"] - truth,
                                 "val_rmse": r["val_rmse"], "fit_seconds": r["fit_seconds"]})
                    print(wid, "short", case, seed, v, round(r["plr"], 3), flush=True)
                pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    if "--short" in sys.argv:
        short_records()
    else:
        main()
    sys.stdout.flush()
    os._exit(0)

"""Information-equal variants on the 240 exact-rate records of the 20 weather sites (protocol P3, committed
2026-10-02 before the runs): kappa-HYDRA-D with g trained on the first 24 data months (rho = 0) and then frozen,
and XGBoost and TCN trained on the whole record (every fifth ISO week held out). Same generator, rates, seeds and
weather as run_semisynthetic_climate.py. Resumable; METHODS, SHARD, RUN_TAG.
Output: results/climate_expansion/info_equal_runs*.csv
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
import run_semisynthetic_climate as climate  # noqa: E402
from run_semisynthetic import G_DATASHEET, synthesize  # noqa: E402

OUT = ROOT / "results/climate_expansion"
METHODS = ("khd_24", "xgboost_full", "tcn_full")


def done_keys() -> set:
    parts = [pd.read_csv(p) for p in OUT.glob("info_equal_runs*.csv")]
    if not parts:
        return set()
    d = pd.concat(parts).dropna(subset=["plr"])
    return set(zip(d.site_id, d.family, d.rate.astype(float), d.seed.astype(int), d.method))


def main(methods: tuple[str, ...]) -> None:
    from src.degradation import ml_plr, perf_models as pm
    from src.degradation.kappa_hydra_d import estimate
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    t = climate.sites()
    t = t[(t.era5 == "ok") & ~t.climate.astype(str).str.startswith("pending")]
    rows = [r for _, r in t.iterrows()]
    shard = os.environ.get("SHARD", "")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        rows = [r for k, r in enumerate(rows) if k % n == i]
    f = OUT / f"info_equal_runs{os.environ.get('RUN_TAG', '')}.csv"
    for row in rows:
        frame, meta = climate.weather(row)
        for family in climate.FAMILIES:
            for rate in climate.RATES:
                for seed in climate.SEEDS:
                    keys = done_keys()
                    todo = [m for m in methods if (row.site_id, family, float(rate), seed, m) not in keys]
                    if not todo:
                        continue
                    syn = synthesize(frame, meta, family, rate, seed)
                    d = None
                    for m in todo:
                        if m == "khd_24":
                            plr = estimate(syn, climate.DC_KW, G_DATASHEET, seeds=(seed,), jackknife=False,
                                           correction_months=24, **hp["kappa_hydra_d"]["params"])["plr"]
                        else:
                            d = d or ml_plr.prepare(syn, climate.DC_KW, full_record=True)
                            base = m.replace("_full", "")
                            pred, _ = pm.fit_predict(base, hp[base]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                                     (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                            plr = ml_plr.plr_from_prediction(syn, climate.DC_KW, pred, d["stamps"])["plr"]
                        out = {"site_id": row.site_id, "climate": row.climate, "family": family, "rate": rate,
                               "seed": seed, "method": m, "plr": plr, "error": plr - rate}
                        cols = list(pd.read_csv(f, nrows=0).columns) if f.exists() else list(out)
                        pd.DataFrame([out]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)
                        print(row.site_id, family, rate, seed, m, round(plr, 3), flush=True)


if __name__ == "__main__":
    sel = os.environ.get("METHODS", "")
    main(tuple(sel.split(",")) if sel else METHODS)
    sys.stdout.flush()
    os._exit(0)

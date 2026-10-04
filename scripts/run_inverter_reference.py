"""Physical reference variants with an inverter model and with coefficients learned on the same record
(protocol amendment 4, committed before the runs).

Measured records (18 main systems, AC power, fixed hour mask, injected rates of P1):
  reference_inverter  expected AC power = PVWatts inverter part-load efficiency curve (pvlib.inverter.pvwatts,
                      nominal efficiency 0.96, inverter DC rating = array DC rating) applied to the PVWatts DC power
                      with the technology-typical temperature coefficient
  reference_physical  expected AC power = a x inverter curve( P_r POA/1000 (1 + gamma (T_c - 25)) (1 + k ln(POA/1000)) ),
                      with a, gamma and the low-light coefficient k fitted (least squares) to the first 12 months of the
                      same record (hours of the fixed mask, POA 200-1200 W/m2); refitted on every injected record
Exact-rate records (20 weather sites; PVWatts and thin-film-like generators, 240 records; ADR generator, 120 records;
generator output is DC, so no inverter curve):
  reference_physical_dc  a, gamma, k fitted to the first 12 months; 'reference' (typical coefficient) is rerun on the
                      same records for a paired comparison. Truth: first-year rate (Eq. 3).
All variants use the year-on-year estimator of the reference.

  python scripts/run_inverter_reference.py measured   -> results/fixed_mask/inverter_reference_runs.csv
  python scripts/run_inverter_reference.py exact      -> results/inverter_reference/exact_runs{RUN_TAG}.csv (SHARD=i/n)
  python scripts/run_inverter_reference.py summary    -> results/inverter_reference/tables/K2_*.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.degradation.rdtools_pipeline import cell_temperature, yoy_from_normalized  # noqa: E402
from src.degradation.truth_convention import first_year  # noqa: E402

OUT = ROOT / "results/inverter_reference"
ETA_NOM, ETA_REF = 0.96, 0.9637


def inverter(pdc: np.ndarray, pdc0: float) -> np.ndarray:
    return np.asarray(pvlib.inverter.pvwatts(pd.Series(pdc).clip(lower=0), pdc0, ETA_NOM, ETA_REF), float)


def dc_model(poa, tc, p_r, gamma, k):
    g = np.clip(poa, 1, None)
    return p_r * g / 1000 * (1 + gamma * (tc - 25)) * (1 + k * np.log(g / 1000))


def fit_physical(frame: pd.DataFrame, dc_kw: float, gamma0: float, mask: pd.Series | None, ac: bool) -> tuple:
    poa = frame.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, frame.get("t_amb_c"), frame.get("t_module_c"), frame.get("wind_ms"))
    first = frame.index < frame.loc[frame.power_w.notna() & (poa > 0)].index.min() + pd.DateOffset(months=12)
    sel = first & poa.between(200, 1200) & frame.power_w.notna() & tc.notna()
    if mask is not None:
        sel &= mask.reindex(frame.index, fill_value=False).astype(bool)
    p_r = dc_kw * 1000.0
    g, t, y = poa[sel].to_numpy(), tc[sel].to_numpy(), frame.power_w[sel].to_numpy()

    def resid(x):
        dc = dc_model(g, t, p_r, x[1], x[2])
        pred = x[0] * (inverter(dc, p_r) if ac else dc)
        return (pred - y) / p_r

    fit = least_squares(resid, x0=[1.0, gamma0, 0.0], bounds=([0.3, -0.008, -0.2], [1.7, 0.0, 0.2]), loss="soft_l1")
    return tuple(float(v) for v in fit.x), int(sel.sum())


def plr_physical(frame: pd.DataFrame, dc_kw: float, a: float, gamma: float, k: float, ac: bool,
                 mask: pd.Series | None) -> dict:
    data = frame[~frame.index.duplicated()].sort_index()
    poa = data.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, data.get("t_amb_c"), data.get("t_module_c"), data.get("wind_ms"))
    p_r = dc_kw * 1000.0
    dc = dc_model(poa.to_numpy(), tc.to_numpy(), p_r, gamma, k)
    exp = a * (inverter(dc, p_r) if ac else dc)
    expected = pd.Series(exp, index=data.index).where(poa > 0)
    power = data.power_w.clip(lower=0)
    return yoy_from_normalized(power / expected.replace(0, np.nan), poa, tc, power, mask=mask)


def measured() -> None:
    import run_fixed_mask as fm
    f = fm.OUT / "inverter_reference_runs.csv"
    done = set(map(tuple, pd.read_csv(f)[["system_id", "injection", "method"]].astype(str).to_numpy())) if f.exists() else set()
    for sid in fm.systems():
        for inj in fm.INJECTIONS:
            base, meta, gamma, ym, _ = fm.load(sid)
            frame = fm.injected(base, inj)
            for method in ("reference_inverter", "reference_physical"):
                if (sid, str(inj), method) in done:
                    continue
                if method == "reference_inverter":
                    a, g, k, n_fit = 1.0, gamma, 0.0, 0
                else:
                    (a, g, k), n_fit = fit_physical(frame, meta["dc_kw"], gamma, ym, ac=True)
                r = plr_physical(frame, meta["dc_kw"], a, g, k, ac=True, mask=ym)
                out = {"system_id": sid, "injection": inj, "method": method, "plr": r["plr"], "ci_low": r["ci_low"],
                       "ci_high": r["ci_high"], "t_center_years": r.get("t_center_years"), "a": a, "gamma": g, "k": k,
                       "fit_hours": n_fit}
                pd.DataFrame([out]).to_csv(f, mode="a", header=not f.exists(), index=False)
                print({x: (round(v, 4) if isinstance(v, float) else v) for x, v in out.items()}, flush=True)


def exact() -> None:
    import run_semisynthetic_climate as climate
    from run_climate_fair_reference import synthesize_adr
    from run_semisynthetic import G_DATASHEET, synthesize
    from src.degradation.rdtools_pipeline import sensor_plr
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"exact_runs{os.environ.get('RUN_TAG', '')}.csv"
    parts = [pd.read_csv(p) for p in OUT.glob("exact_runs*.csv")]
    done = set(map(tuple, pd.concat(parts)[["site_id", "family", "rate", "seed", "method"]].astype(str).to_numpy())) if parts else set()
    table = climate.sites()
    table = table[(table.era5 == "ok") & ~table.climate.astype(str).str.startswith("pending")]
    rows = [r for _, r in table.iterrows()]
    shard = os.environ.get("SHARD", "")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        rows = [r for k, r in enumerate(rows) if k % n == i]
    for row in rows:
        frame, meta = climate.weather(row)
        for family in ("pvwatts", "thin_film", "adr"):
            for rate in (-0.5, -1.0):
                for seed in (0, 1, 2):
                    todo = [m for m in ("reference", "reference_physical_dc")
                            if (row.site_id, family, str(rate), str(seed), m) not in done]
                    if not todo:
                        continue
                    syn = synthesize_adr(frame, rate, seed) if family == "adr" else synthesize(frame, meta, family, rate, seed)
                    truth = float(first_year(rate))
                    for m in todo:
                        if m == "reference":
                            r, (a, g, k) = sensor_plr(syn, 5.0, G_DATASHEET), (1.0, G_DATASHEET, 0.0)
                        else:
                            (a, g, k), _ = fit_physical(syn, 5.0, G_DATASHEET, None, ac=False)
                            r = plr_physical(syn, 5.0, a, g, k, ac=False, mask=None)
                        out = {"site_id": row.site_id, "climate": row.climate, "country": row.country, "family": family,
                               "rate": rate, "seed": seed, "method": m, "plr": r["plr"], "truth": truth,
                               "error": r["plr"] - truth, "a": a, "gamma": g, "k": k}
                        pd.DataFrame([out]).to_csv(f, mode="a", header=not f.exists(), index=False)
                        print(row.site_id, family, rate, seed, m, round(r["plr"], 3), flush=True)


def summary() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    d = pd.concat([pd.read_csv(p) for p in OUT.glob("exact_runs*.csv")]).dropna(subset=["plr"])
    d = d.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    d["group"] = np.where(d.family == "adr", "adr", "pvwatts_thin_film")
    rows = []
    for (grp, fam, m), g in pd.concat([d.assign(fam="all"), d.assign(fam=d.family)]).groupby(["group", "fam", "method"]):
        site = g.groupby("site_id").error.agg(lambda e: e.abs().mean())
        rows.append({"group": grp, "family": fam, "method": m, "n": len(g), "mae": site.mean(), "worst_site": site.max(),
                     "worst_site_id": site.idxmax(), "bias": g.error.mean(), "median_k": g.k.median(),
                     "median_gamma": g.gamma.median()})
    pd.DataFrame(rows).to_csv(OUT / "tables/K2_exact_rate.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    {"measured": measured, "exact": exact}.get(sys.argv[1] if len(sys.argv) > 1 else "summary", summary)()
    sys.stdout.flush()
    os._exit(0)

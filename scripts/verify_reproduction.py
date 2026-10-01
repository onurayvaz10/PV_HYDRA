"""Independent reproduction check of the shipped results (for reviewers).

A. Summaries: every result table is regenerated from the raw run files shipped in results/degradation_paper
   (summarize_degradation, summarize_robustness, summarize_forward_validation, robustness_checks) and compared with the
   shipped copy, cell by cell. Tables that need the hourly validation predictions (T3b, T4; not shipped because they
   embed measured power) are skipped when those are absent.
B. Re-execution: a sample of the revision experiments is rerun from the open PVDAQ records (03_Datasets) and ERA5:
   - check_injection_truth: every Henderson (PVDAQ 1423) record (reference workflow, deterministic);
   - run_identifiability: the reference and κ-HYDRA-D on four Arbuckle (PVDAQ 2107) records with and without weather
     drift (κ-HYDRA-D on CPU or GPU; tolerance 0.02 %/yr for floating-point differences between devices).
C. The number ledger of the paper (every printed number recomputed from the result tables).
Output: results/REPRODUCTION_REPORT.txt; exit code 0 only if every check passes.

  python scripts/verify_reproduction.py            # A + B + C
  python scripts/verify_reproduction.py --quick    # A + C only (no raw data needed)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TAB = ROOT / "results/degradation_paper/tables"
RES = ROOT / "results/degradation_paper"
LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LINES.append(msg)


def compare_tables(before: Path) -> bool:
    ok = True
    for f in sorted(before.glob("T*.csv")):
        new = TAB / f.name
        if not new.exists():
            log(f"  MISSING after regeneration: {f.name}")
            ok = False
            continue
        a, b = pd.read_csv(f), pd.read_csv(new)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            log(f"  SHAPE/COLUMNS differ: {f.name} {a.shape} vs {b.shape}")
            ok = False
            continue
        worst = 0.0
        for col in a.columns:
            num = lambda x: pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)  # noqa: E731
            if num(a[col]) and num(b[col]):
                d = (a[col] - b[col]).abs()
                d = d[~(a[col].isna() & b[col].isna())]
                worst = max(worst, float(d.max()) if len(d) else 0.0)
            elif not a[col].astype(str).equals(b[col].astype(str)):
                worst = np.inf
        status = "identical" if worst < 1e-9 else ("OK (<1e-6)" if worst < 1e-6 else f"DIFFERS (max {worst:.3g})")
        ok &= worst < 1e-6
        log(f"  {f.name:42s} {status}")
    return ok


def part_a() -> bool:
    log("A. Regenerating the summary tables from the shipped raw run files")
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(TAB, tmp / "tables")
    vp = RES / "val_preds"
    if not vp.exists():             # T3b and T4 need the hourly validation predictions (not shipped)
        log("  (hourly validation predictions not present: T3b/T4 are kept as shipped)")
    for script in ("summarize_degradation.py", "summarize_robustness.py", "summarize_forward_validation.py",
                   "robustness_checks.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        log(f"  ran {script}: exit {r.returncode}")
        if r.returncode:
            log(r.stderr[-2000:])
            return False
    return compare_tables(tmp / "tables")


def part_b() -> bool:
    log("B. Re-executing a sample of the revision experiments from the open PVDAQ records")
    import check_injection_truth as cit
    import run_identifiability as ri
    import run_semisynthetic as semi
    from src.degradation import kappa_hydra_d as kd
    from src.degradation import ml_plr
    from src.degradation.rdtools_pipeline import sensor_plr
    from src.degradation.tier_a import load_system
    ok = True
    t26 = pd.read_csv(TAB / "T26_injection_truth_check.csv")
    frame, meta = load_system("1423")
    for d_true in (0.0, -0.8, -2.0):
        base = cit.record(frame, d_true)
        b = sensor_plr(base, 5.0, cit.G)
        for r in (-0.5, -1.0, -2.0):
            obs = sensor_plr(base.assign(power_w=ml_plr.inject(base, r)), 5.0, cit.G)["plr"] - b["plr"]
            ref = t26[(t26.weather == "pvdaq_1423") & (t26.d_true == d_true) & (t26.r == r)].observed_change.iloc[0]
            good = abs(obs - ref) < 1e-6
            ok &= good
            log(f"  injection truth 1423 d={d_true:+.1f} r={r:+.1f}: rerun {obs:.5f} stored {ref:.5f} {'OK' if good else 'DIFF'}")
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    stored = pd.read_csv(RES / "identifiability_runs.csv").drop_duplicates(
        ["weather", "family", "rate", "seed", "drift", "method"], keep="last")
    frame, meta = load_system("2107")
    for drift in ("none", "drift"):
        fr = ri.drifted(frame) if drift == "drift" else frame
        for rate in (0.0, -1.0):
            syn = semi.synthesize(fr, meta, "thin_film", rate, 0)
            new = {"rdtools": sensor_plr(syn, 5.0, semi.G_DATASHEET)["plr"],
                   "khd_full": kd.estimate(syn, 5.0, semi.G_DATASHEET, seeds=(0,), jackknife=False, **hp)["plr"]}
            for m, v in new.items():
                ref = stored[(stored.weather == "pvdaq_2107") & (stored.family == "thin_film") & (stored.rate == rate)
                             & (stored.seed == 0) & (stored.drift == drift) & (stored.method == m)].plr.iloc[0]
                tol = 1e-6 if m == "rdtools" else 0.02
                good = abs(v - ref) < tol
                ok &= good
                log(f"  identifiability 2107 {drift:5s} r={rate:+.1f} {m:9s}: rerun {v:+.4f} stored {ref:+.4f} "
                    f"(tol {tol:g}) {'OK' if good else 'DIFF'}")
    return ok


def part_c() -> bool:
    log("C. Number ledger of the paper (paper/result_ledger.csv recomputed from the result tables)")
    before = (ROOT / "paper/result_ledger.csv").read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "paper/scripts/build_result_ledger.py")], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    after = (ROOT / "paper/result_ledger.csv").read_text(encoding="utf-8")
    log(f"  build_result_ledger.py exit {r.returncode}; ledger {'identical' if before == after else 'CHANGED'}")
    return r.returncode == 0 and before == after


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import platform
    import torch
    log(f"Python {platform.python_version()} | {platform.platform()} | device "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    results = {"A summaries": part_a()}
    if "--quick" not in sys.argv:
        results["B re-execution"] = part_b()
    results["C ledger"] = part_c()
    log("RESULT: " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    (ROOT / "results/REPRODUCTION_REPORT.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

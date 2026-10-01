"""Run manifest for reproducibility (H15): SHA-256, size, rows and producing script of every result table, plus the
software environment, seeds and data identifiers. Output: results/RUN_MANIFEST.csv, results/RUN_ENVIRONMENT.json"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRODUCER = {"T0_": "make_site_table.py", "T0b": "make_site_map.py", "T1_": "summarize_degradation.py",
            "T2_": "summarize_degradation.py", "T3": "summarize_degradation.py", "T4": "summarize_degradation.py",
            "T5": "summarize_degradation.py", "T6": "summarize_degradation.py", "T7": "benchmark_cost.py",
            "T8": "summarize_degradation.py", "T9": "summarize_degradation.py", "T10": "run_bin_diagnostic.py",
            "T11": "summarize_degradation.py", "T12": "run_refwindow_sensitivity.py", "T13": "run_loss_structure.py",
            "T14": "test_loss_structure.py", "T15": "test_loss_structure.py", "T16": "summarize_semisynthetic_ci.py",
            "T17": "summarize_semisynthetic_ci.py", "T18": "summarize_robustness.py", "T19": "summarize_robustness.py",
            "T20": "summarize_robustness.py", "T21": "summarize_robustness.py", "T22": "summarize_forward_validation.py",
            "T23": "summarize_forward_validation.py", "T24": "summarize_forward_validation.py",
            "T25": "summarize_robustness.py", "T26": "check_injection_truth.py", "T27": "robustness_checks.py",
            "T28": "robustness_checks.py", "T29": "robustness_checks.py"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for p in sorted((RES / "degradation_paper").rglob("*.csv")):
        if "val_preds" in p.parts or p.name.startswith("_"):
            continue
        try:
            n = len(pd.read_csv(p))
        except Exception:
            n = None
        prod = next((v for k, v in PRODUCER.items() if p.name.startswith(k)), "run script named in README (step table)")
        rows.append({"file": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size, "rows": n,
                     "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                     "produced_by": prod})
    pd.DataFrame(rows).to_csv(RES / "RUN_MANIFEST.csv", index=False)
    pkgs = {}
    for name in ("numpy", "pandas", "torch", "xgboost", "scikit-learn", "optuna", "rdtools", "pvlib", "shap", "scipy",
                 "statsmodels", "rasterio", "python-docx", "openpyxl"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda = torch.version.cuda
    except Exception:
        gpu, cuda = "unknown", None
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    env = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": sys.version.split()[0],
           "platform": platform.platform(), "gpu": gpu, "cuda": cuda, "packages": pkgs,
           "seeds": {"performance models": [0, 1, 2, 3, 4], "semi-synthetic noise": [0, 1, 2], "optuna sampler": 42,
                     "identifiability test": [0, 1]},
           "tuning": {m: {"trials": v.get("trials"), "hpo_seconds": v.get("hpo_seconds")} for m, v in hp.items()},
           "data": {"DKASC": "dkasolarcentre.com.au export files, SHA-256 in data/raw/external/dkasc/manifest.json",
                    "PVDAQ": "OEDI doi:10.25984/1846021, catalog 2025-07-29",
                    "ERA5": "Open-Meteo archive (doi:10.5281/zenodo.7970649), cached responses with SHA-256",
                    "Koppen-Geiger": "Beck et al. 2023, figshare 10.6084/m9.figshare.21789074.v2, raster SHA-256 in data/reference/climate/source.json"},
           "result_files": len(rows)}
    (RES / "RUN_ENVIRONMENT.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"{len(rows)} result files; environment written")


if __name__ == "__main__":
    main()

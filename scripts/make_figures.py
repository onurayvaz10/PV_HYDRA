"""Paper figures in IEEE Transactions format, from result files only.

IEEE rules applied: one-column width (3.5 in / 88 mm), Arial 8 pt text (legible 8-10 pt after
placement), axis labels as words with units in parentheses, no caption inside the figure, colour
figures at 600 dpi PNG plus a vector PDF with embedded fonts. Colour-blind-safe Okabe-Ito palette,
distinguishable in grey scale by marker/hatch. Each figure is skipped when its inputs are missing.
Output: reports/manuscript/figures/

F1 irradiance-sensor drift (4 systems, ERA5 and clear-sky references)
F2 PLR per system: RdTools 95 % CI, two-stage models (seed mean), kappa-HYDRA-D with CI
F3 known-truth recovery error by method (all injected rates)
F4 kappa-HYDRA-D ablation: recovery error and seed spread by variant
F5 transfer for young plants
F6 SHAP shares of physical inputs per system
F7 semi-synthetic records with an exact rate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results/degradation_paper"
TAB = RES / "tables"
FIG = ROOT / "reports/manuscript/figures"
COL = 3.5                     # IEEE one-column width (in)
PAL = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
MODELS = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
NAMES = {"rdtools": "RdTools", "xgboost": "XGBoost", "random_forest": "RF", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN",
         "patchtst": "PatchTST", "informer": "Informer", "khd_full": "κ-HYDRA-D", "khd_no_physics": "No physics",
         "khd_no_correction": "No correction", "khd_time_input": "Time input"}
FAMILY = {"rdtools": PAL[0], "xgboost": PAL[1], "random_forest": PAL[1], "lstm": PAL[2], "gru": PAL[2], "tcn": PAL[2],
          "patchtst": PAL[5], "informer": PAL[5], "khd_full": PAL[6]}
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "pdf.fonttype": 42})


def save(fig, name, rect=None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3, rect=rect)          # rect reserves room for a figure-level legend
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def f1_drift():
    cache = RES / "drift_series.csv"
    if not cache.exists():
        from src.degradation.tier_a import clearsky_drift, load_system, sensor_drift
        rows = []
        for sid in ("dkasc_7", "1430", "1432", "1433"):
            frame, meta = load_system(sid)
            for ref, fn in (("ERA5", sensor_drift), ("clear-sky", clearsky_drift)):
                s = fn(frame, meta, series=True)["_series"]
                rows += [{"system": meta["system_id"], "reference": ref, "month": str(p), "anomaly": v} for p, v in s.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)
    d = pd.read_csv(cache)
    d["t"] = pd.PeriodIndex(d.month, freq="M").to_timestamp()
    screen = pd.read_csv(RES / "tier_a_screen.csv").set_index("system_id")
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.8), sharey=True)
    for ax, (sid, g) in zip(axes.flat, d.groupby("system", sort=False)):
        for (ref, h), c, mk in zip(g.groupby("reference"), (PAL[5], PAL[6]), (".", "x")):
            ax.plot(h.t, h.anomaly, mk, ms=1.8, color=c, alpha=0.55, mew=0.5)
            x = (h.t - h.t.min()).dt.days / 365.25
            k = np.polyfit(x, h.anomaly, 1)
            ax.plot(h.t, np.polyval(k, x), color=c, lw=1.1, label=f"{ref}: {100 * k[0]:+.2f} %/yr")
        status = "excluded" if sid in screen.index and not screen.loc[sid, "pass"] else "retained"
        ax.set_title(f"{sid.replace('pvdaq_', 'PVDAQ ').replace('dkasc_', 'DKASC ')} ({status})", pad=2)
        ax.axhline(1, color="0.7", lw=0.5)
        ax.set_ylim(0.7, 1.4)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.legend(frameon=False, loc="lower left", handlelength=1.2, borderaxespad=0.1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized POA ratio")
    save(fig, "F1_sensor_drift")


def f2_plr():
    t1 = TAB / "T1_plr_by_system.csv"
    if not t1.exists():
        return
    d = pd.read_csv(t1).sort_values("rdtools_plr").reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(COL, 0.175 * len(d) + 1.05))
    ax.errorbar(d.rdtools_plr, y, xerr=[d.rdtools_plr - d.rdtools_ci_low, d.rdtools_ci_high - d.rdtools_plr],
                fmt="s", color=PAL[0], ms=3, capsize=1.5, lw=0.8, label="RdTools YoY (95 % CI)")
    markers = "o^vD<>p"
    for i, m in enumerate(MODELS):
        col = f"{m}_plr_mean"
        if col in d:
            ax.plot(d[col], y + 0.1 * (i - 3) * 0.6, markers[i], ms=2.3, color=PAL[(i + 1) % len(PAL)], mew=0.3,
                    label=NAMES[m], alpha=0.9)
    if "khd_plr" in d:
        ax.errorbar(d.khd_plr, y - 0.32, xerr=[d.khd_plr - d.khd_ci_low, d.khd_ci_high - d.khd_plr], fmt="*",
                    color=PAL[6], ms=4, capsize=1.5, lw=0.8, label="κ-HYDRA-D (95 % CI)")
    ax.set_yticks(y, [f"{s.replace('pvdaq_', 'PVDAQ ').replace('dkasc_', 'DKASC ')} ({c})" for s, c in zip(d.system_id, d.climate)],
                  fontsize=6.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlabel("Performance loss rate (%/yr)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.12 - 0.3 / len(d)), columnspacing=0.8,
              handletextpad=0.3)
    save(fig, "F2_plr_by_system")


def f3_recovery():
    f = TAB / "T2_recovery_by_system.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    order = [m for m in ["rdtools", *MODELS, "khd_full"] if m in set(d.method)]
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    bp = ax.boxplot([d[d.method == m].error for m in order], widths=0.6, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 0.8}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
    for box, m in zip(bp["boxes"], order):
        box.set_facecolor(FAMILY.get(m, PAL[8]))
        box.set_alpha(0.75)
        box.set_linewidth(0.6)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(range(1, len(order) + 1), [NAMES.get(m, m) for m in order], rotation=35, ha="right")
    ax.set_ylabel("Estimated − expected change (%/yr)")
    save(fig, "F3_known_truth_recovery")


def f4_ablation():
    f = TAB / "T2_recovery_by_system.csv"
    kd_f = RES / "kappa_hydra_d_runs.csv"
    if not (f.exists() and kd_f.exists()):
        return
    d = pd.read_csv(f)
    d = d[d.method.str.startswith("khd_") & (d.rate == -1.0)]
    kd = pd.read_csv(kd_f)
    kd = kd[kd.injection == 0]
    variants = [v for v in ("full", "no_physics", "no_correction", "time_input") if f"khd_{v}" in set(d.method)]
    if not variants:
        return
    labels = [NAMES[f"khd_{v}"].replace(" ", "\n") for v in variants]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.9))
    a.boxplot([np.abs(d[d.method == f"khd_{v}"].error) for v in variants], showfliers=False, widths=0.6)
    a.set_xticks(range(1, len(variants) + 1), labels, fontsize=6)
    a.set_ylabel("Absolute recovery error (%/yr)")
    b.boxplot([kd[kd.variant == v].plr_seed_sd for v in variants], showfliers=False, widths=0.6)
    b.set_xticks(range(1, len(variants) + 1), labels, fontsize=6)
    b.set_ylabel("Seed SD of rate (%/yr)")
    save(fig, "F4_khd_ablation")


def f5_transfer():
    f = TAB / "T9_transfer_by_system.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    conds = [c for c in ("scratch_24", "scratch_6", "finetune_6", "zero_shot") if c in set(d.condition)]
    names = {"scratch_24": "24\nmo.", "scratch_6": "6\nmo.", "finetune_6": "PT +\n6 mo.",   # PT: pre-trained (caption)
             "zero_shot": "Zero-\nshot"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 2.2))
    for i, m in enumerate(sorted(d.model.unique())):
        g = d[d.model == m].groupby("condition")
        x = np.arange(len(conds)) + (i - 0.5) * 0.36
        a.bar(x, [np.abs(g.get_group(c).recovery_error).mean() for c in conds], 0.36, color=PAL[i + 1], label=NAMES.get(m, m),
              hatch=("" if i == 0 else "///"), edgecolor="black", lw=0.4)
        b.bar(x, [g.get_group(c).abs_diff_rdtools.mean() for c in conds], 0.36, color=PAL[i + 1],
              hatch=("" if i == 0 else "///"), edgecolor="black", lw=0.4)
    for ax, lab in ((a, "Mean abs. recovery error (%/yr)"), (b, "Difference from RdTools (%/yr)")):
        ax.set_xticks(np.arange(len(conds)), [names[c] for c in conds], fontsize=6.5)
        ax.set_ylabel(lab)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "F5_transfer", rect=(0, 0, 1, 0.9))


def f6_shap():
    f = RES / "shap_by_system.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    parts = {"POA": ["shap_poa_n"], "Cell temperature": ["shap_tcell_n"], "Ambient temperature": ["shap_tamb_n"],
             "Hour of day": ["shap_hour_sin", "shap_hour_cos"], "Day of year": ["shap_doy_sin", "shap_doy_cos"]}
    hatches = ["", "////", "....", "xxxx", "\\\\\\\\"]
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    bottom = np.zeros(len(d))
    for i, (lab, cols) in enumerate(parts.items()):
        v = d[cols].sum(axis=1).to_numpy()
        ax.bar(np.arange(len(d)), v, bottom=bottom, color=PAL[i + 1], label=lab, hatch=hatches[i], edgecolor="white", lw=0.2)
        bottom += v
    ax.set_xticks(np.arange(len(d)), [s.replace("pvdaq_", "PVDAQ ").replace("dkasc_", "DKASC ") for s in d.system_id],
                  rotation=90, fontsize=5.5)
    ax.set_ylabel("Share of mean |SHAP value|")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), columnspacing=0.8)
    save(fig, "F6_shap_drivers")


def f7_semisynthetic():
    f = RES / "semisynthetic_runs.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    methods = [m for m in ("rdtools", "xgboost", "tcn", "khd_full") if m in set(d.method)]
    fams = [x for x in ("pvwatts", "thin_film") if x in set(d.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(COL, 2.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        g = d[d.family == fam]
        bp = ax.boxplot([g[g.method == m].error for m in methods], widths=0.6, patch_artist=True,
                        flierprops={"ms": 2}, medianprops={"color": "black", "lw": 0.8})
        for box, m in zip(bp["boxes"], methods):
            box.set_facecolor(FAMILY.get(m, PAL[8]))
            box.set_alpha(0.75)
        ax.set_xticks(range(1, len(methods) + 1), [NAMES.get(m, m) for m in methods], rotation=35, ha="right")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title({"pvwatts": "(a) PVWatts response", "thin_film": "(b) Thin-film-like response"}[fam], pad=2)
    axes[0].set_ylabel("Rate error (%/yr)")
    save(fig, "F7_semisynthetic")


if __name__ == "__main__":
    for fn in (f1_drift, f2_plr, f3_recovery, f4_ablation, f5_transfer, f6_shap, f7_semisynthetic):
        try:
            fn()
        except Exception as exc:  # one failing figure must not block the others; reported
            print(fn.__name__, "failed:", type(exc).__name__, exc, flush=True)
    sys.stdout.flush()
    os._exit(0)

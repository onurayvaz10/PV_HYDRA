"""Figures 1-3 of the paper, regenerated from the result files (no values read from images).

Fig. 1  (a) the five measured locations of the main benchmark (18 systems) and the 20 weather sites of the exact-rate
        records on the 1-km Koppen-Geiger map (Beck et al. 2023); (b) structure of kappa-HYDRA-D.    181 mm
Fig. 3  (a) injected-change error with the fixed hour mask, unit = system x injected rate
        (results/fixed_mask/tables/R0_injection_rows.csv); (b) exact-rate absolute error, unit = weather site
        (semisynthetic_climate_runs*.csv, info_equal_runs*.csv). Colour and hatch = method family; dashed line =
        practical-equivalence margin delta = 0.05 %/yr.                                                   88 mm
Fig. 2  measured main benchmark, systems sorted by technology: (a) reference, seven two-stage and kappa-HYDRA-D
        rates on the unmodified records; (b) reference-window range of the XGBoost rate (refwindow_fixed.csv).

Okabe-Ito palette with hatch patterns (readable in grey scale), 8 pt text at final size, panel labels (a), (b).
Output: paper/figures/Fig{1,2,3}.pdf (vector) and .png (600 dpi).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

REV = Path(__file__).resolve().parents[1]          # paper/
ROOT = REV.parent
sys.path.insert(0, str(ROOT))
FM = ROOT / "results/fixed_mask"
EXP = ROOT / "results/climate_expansion"
OUT = REV / "figures"
MM = 1 / 25.4
DELTA = 0.05
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
# method key in the injection table, key in the exact-rate runs, label, colour, hatch
METHODS = [("reference", "rdtools", "Reference", OI["grey"], ""),
           ("khd", "khd_full", "κ-HYDRA-D", OI["verm"], "////"),
           ("khd_24", "khd_24", "κ-HYDRA-D-24", OI["verm"], "...."),
           ("xgboost", "xgboost", "XGBoost", OI["blue"], "\\\\\\\\"),
           ("random_forest", "random_forest", "Random forest", OI["blue"], "\\\\\\\\"),
           ("lstm", "lstm", "LSTM", OI["green"], "xxx"),
           ("gru", "gru", "GRU", OI["green"], "xxx"),
           ("tcn", "tcn", "TCN", OI["orange"], "---"),
           ("patchtst", "patchtst", "PatchTST", OI["purple"], "++"),
           ("informer", "informer", "Informer", OI["purple"], "++")]
SEVEN = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.dpi": 600, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.5})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def read(pattern: str, base: Path) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(base.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------------------------- Fig. 1
MAIN = pd.DataFrame([("Alice Springs", -23.7618, 133.8748, 12), ("Golden", 39.7421, -105.1776, 3),
                     ("Henderson", 36.0275, -114.9215, 1), ("Las Vegas", 36.1952, -115.1582, 1),
                     ("Arbuckle", 38.9963, -122.1341, 1)], columns=["location", "lat", "lon", "systems"])


def box(ax, x, y, w, h, fc, text, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.015", fc=fc,
                                ec="#333333", lw=0.6))
    lines = text.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        yy = y + h / 2 + (n - 1) * 0.0255 - i * 0.051
        ax.text(x + w / 2, yy, ln, ha="center", va="center", fontsize=8,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333",
                                 shrinkA=0, shrinkB=0))


def fig1() -> None:
    import rasterio
    from matplotlib.colors import ListedColormap
    from rasterio.enums import Resampling
    from scripts.make_world_map import CLIM, legend
    lut = legend()
    with rasterio.open(CLIM / "koppen_geiger_1991_2020_0p00833333.tif") as src:
        data = src.read(1, out_shape=(1080, 2160), resampling=Resampling.mode)
        b = src.bounds
    codes = np.arange(0, max(lut) + 1)
    cmap = ListedColormap([(1, 1, 1)] + [lut.get(c, ("", (1, 1, 1)))[1] for c in codes[1:]])
    semi = pd.read_csv(EXP / "semisynthetic_sites.csv")

    fig = plt.figure(figsize=(181 * MM, 76 * MM))
    ax = fig.add_axes([0.075, 0.25, 0.55, 0.68])
    ax.imshow(np.ma.masked_equal(data, 0), cmap=cmap, vmin=0, vmax=len(codes) - 1, alpha=0.35,
              extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
    ax.scatter(semi.longitude, semi.latitude, marker="^", s=26, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
    ax.scatter(MAIN.lon, MAIN.lat, marker="o", s=22 + 9 * np.sqrt(MAIN.systems), facecolor=OI["verm"],
               edgecolor="black", lw=0.6, zorder=4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-50, -25, 0, 25, 50, 75])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.text(-0.13, 1.03, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", mfc=OI["verm"], mec="black", ms=6,
                      label=f"Main benchmark: {int(MAIN.systems.sum())} measured systems, {len(MAIN)} locations"),
               Line2D([], [], marker="^", ls="", mfc="white", mec="black", ms=6,
                      label=f"Weather site of the exact-rate records ({len(semi)})")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=1, frameon=False,
               handletextpad=0.3, labelspacing=0.35)

    d = fig.add_axes([0.645, 0.0, 0.355, 1.0])
    d.set_xlim(0, 1)
    d.set_ylim(0, 1)
    d.axis("off")
    d.text(0.0, 1.0, "(b)", ha="left", va="top", fontsize=8, fontweight="bold")
    w, xs = 0.31, (0.01, 0.345, 0.68)
    mid = [x + w / 2 for x in xs]
    c_in, c_net, c_phys, c_tr, c_out = "#e8eef7", "#e2f0d9", "#fde7c8", "#f3e1f0", "#eeeeee"
    box(d, xs[0], 0.835, w, 0.11, c_in, "Weather x\nno time input")
    box(d, xs[1], 0.835, w, 0.11, c_in, "POA, $T_c$\nrating")
    box(d, xs[2], 0.835, w, 0.11, c_in, "Time t\n(years)")
    box(d, xs[0], 0.66, w, 0.13, c_net, "Correction\n$g_{0.5}(x)$")
    box(d, xs[1], 0.66, w, 0.13, c_phys, "PVWatts\n$P_{\\mathrm{PVW}}$")
    box(d, xs[2], 0.66, w, 0.13, c_tr, "Trend\n$1+ρ(t-t_{\\mathrm{mid}})$")
    for m in mid:
        arrow(d, (m, 0.835), (m, 0.79))
    d.scatter([0.5], [0.605], s=110, marker="o", facecolor="white", edgecolor="#333333", lw=0.7, zorder=5)
    d.text(0.5, 0.604, "×", ha="center", va="center", fontsize=9, zorder=6)
    arrow(d, (mid[0], 0.66), (0.47, 0.612))
    arrow(d, (mid[1], 0.66), (0.5, 0.632))
    arrow(d, (mid[2], 0.66), (0.53, 0.612))
    box(d, 0.01, 0.435, 0.98, 0.12, c_out, "Median power\n$\\hat{P} = P_{\\mathrm{PVW}}\\,g_{0.5}(x)\\,[1+ρ(t-t_{\\mathrm{mid}})]$")
    arrow(d, (0.5, 0.578), (0.5, 0.555))
    box(d, 0.01, 0.225, 0.98, 0.17, "#fff6cc", "Joint fit, insolation-weighted loss\nwhole record (κ-HYDRA-D) or\ng on first 24 months (κ-HYDRA-D-24)")
    arrow(d, (0.5, 0.435), (0.5, 0.395))
    box(d, 0.01, 0.015, 0.98, 0.17, c_out, "Output\nfirst-year rate r and\n95 % interval")
    arrow(d, (0.5, 0.225), (0.5, 0.185))
    save(fig, "Fig1")


# ----------------------------------------------------------------------------------------------- Fig. 2
def strip_box(ax, values: list[np.ndarray]) -> None:
    rng = np.random.default_rng(0)
    bp = ax.boxplot(values, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 0.9}, boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6},
                    capprops={"lw": 0.6})
    for patch, (_, _, _, c, h) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(matplotlib.colors.to_rgba(c, 0.45))
        patch.set_hatch(h)
        patch.set_edgecolor("black")
    for i, v in enumerate(values, 1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=2.5, color="black", lw=0, alpha=0.6, zorder=3)


def fig2() -> None:
    rows = pd.read_csv(FM / "tables/R0_injection_rows.csv")
    clim = pd.concat([read("semisynthetic_climate_runs*.csv", EXP), read("info_equal_runs*.csv", EXP)]).dropna(subset=["plr"])
    clim = clim.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    site = clim.groupby(["method", "site_id"]).error.agg(lambda e: e.abs().mean()).reset_index()
    va = [rows[rows.method == k].error.abs().to_numpy() for k, *_ in METHODS]
    vb = [site[site.method == c].error.to_numpy() for _, c, *_ in METHODS]
    n_a = int(rows[rows.method == "reference"].shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(88 * MM, 92 * MM), sharex=True)
    for ax, vals, ylab in ((axes[0], va, "Injected-change error (%/yr)"), (axes[1], vb, "Exact-rate error (%/yr)")):
        strip_box(ax, vals)
        ax.set_yscale("log")
        ax.axhline(DELTA, color="black", lw=0.7, ls="--", zorder=1)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", lw=0.3, color="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(1e-4, 4)
    axes[1].set_ylim(1e-3, 0.5)
    axes[0].text(len(METHODS) + 0.45, DELTA * 1.12, "δ", ha="right", va="bottom")
    axes[1].set_xticks(range(1, len(METHODS) + 1), [m[2] for m in METHODS], rotation=45, ha="right")
    for ax_, l_ in ((axes[0], "a"), (axes[1], "b")):
        ax_.text(0.015, 0.97, f"({l_})", transform=ax_.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    axes[0].text(0.99, 0.97, f"unit: system × rate, n = {n_a}", transform=axes[0].transAxes, ha="right", va="top")
    axes[1].text(0.99, 0.97, "unit: weather site, n = 20", transform=axes[1].transAxes, ha="right", va="top")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.2, hspace=0.18)
    save(fig, "Fig3")      # known-rate errors: third figure of the paper


# ----------------------------------------------------------------------------------------------- Fig. 3
# label, technology group (sort key); PVDAQ technologies are not catalogued except the CIS array
SYSTEMS = {"dkasc_7": ("A7 CdTe", 0), "dkasc_8": ("A8 a-Si", 0), "pvdaq_10": ("G10 CIS", 0),
           "dkasc_17": ("A17 HIT", 1),
           "dkasc_10": ("A10 m-Si", 2), "dkasc_12": ("A12 m-Si", 2), "dkasc_13": ("A13 m-Si", 2),
           "dkasc_18": ("A18 m-Si", 2), "dkasc_19": ("A19 m-Si", 2),
           "dkasc_11": ("A11 p-Si", 3), "dkasc_14": ("A14 p-Si", 3), "dkasc_20": ("A20 p-Si", 3),
           "dkasc_21": ("A21 p-Si", 3),
           "pvdaq_1430": ("G1430 trk", 4), "pvdaq_33": ("G33", 4), "pvdaq_1423": ("Henderson", 4),
           "pvdaq_1278": ("Las Vegas", 4), "pvdaq_2107": ("Arbuckle", 4)}
GROUPS = ["thin film", "HIT", "mono-Si", "poly-Si", "c-Si, type n/a"]


def fig3() -> None:
    rd = read("rd_runs*.csv", FM).drop_duplicates(["system_id", "injection"], keep="last")
    rd = rd[rd.injection == 0].set_index("system_id")
    k = read("kappa_runs*.csv", FM).drop_duplicates(["system_id", "variant", "injection"], keep="last")
    k = k[(k.variant == "full") & (k.injection == 0)].set_index("system_id")
    ml = read("ml_runs*.csv", FM).drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
    ml = ml[ml.injection == 0].groupby(["system_id", "model"]).plr.mean().unstack()
    win = pd.read_csv(FM / "refwindow_fixed.csv").set_index("system_id")
    order = sorted(SYSTEMS, key=lambda s: (SYSTEMS[s][1], SYSTEMS[s][0]))
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(88 * MM, 84 * MM), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
    a = axes[0]
    two = ml.reindex(order)[SEVEN]
    a.vlines(x, two.min(axis=1), two.max(axis=1), color=OI["blue"], lw=3.2, alpha=0.45, zorder=2)
    for m in SEVEN:
        a.scatter(x, two[m], s=6, color=OI["blue"], lw=0, zorder=3)
    r = rd.reindex(order)
    a.errorbar(x - 0.28, r.plr, yerr=[r.plr - r.ci_low, r.ci_high - r.plr], fmt="o", ms=3, color="black", lw=0.6,
               capsize=0, zorder=4)
    kk = k.reindex(order)
    a.errorbar(x + 0.28, kk.plr, yerr=[kk.plr - kk.ci_low, kk.ci_high - kk.plr], fmt="D", ms=3, color=OI["verm"],
               lw=0.6, capsize=0, zorder=4)
    a.axhline(0, color="#999999", lw=0.5)
    a.set_ylabel("Rate (%/yr)")
    a.grid(axis="y", lw=0.3, color="#dddddd")
    a.spines[["top", "right"]].set_visible(False)
    bounds = [i for i in range(1, len(order)) if SYSTEMS[order[i]][1] != SYSTEMS[order[i - 1]][1]]
    for ax_ in axes:
        for bnd in bounds:
            ax_.axvline(bnd - 0.5, color="#bbbbbb", lw=0.5, ls=":")
    fig.legend(handles=[Line2D([], [], marker="o", color="black", ms=3, lw=0.6, label="Reference, 95 % CI"),
                        Line2D([], [], color=OI["blue"], lw=3.2, alpha=0.45, label="Seven two-stage rates"),
                        Line2D([], [], marker="D", color=OI["verm"], ms=3, lw=0.6, label="κ-HYDRA-D, 95 % CI")],
               loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=3, frameon=False, handlelength=1.3,
               columnspacing=0.8, handletextpad=0.3)
    a.text(-0.13, 1.0, "(a)", transform=a.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    bax = axes[1]
    rngw = win.reindex(order).window_range.to_numpy()
    bax.bar(x, rngw, width=0.6, color=OI["orange"], edgecolor="black", lw=0.4, hatch="///")
    med = float(np.nanmedian(rngw))
    bax.axhline(med, color="black", lw=0.7, ls="--")
    bax.text(len(order) - 0.5, med, "median", ha="right", va="bottom")
    bax.set_ylabel("Window range (%/yr)")
    bax.spines[["top", "right"]].set_visible(False)
    bax.set_xticks(x, [SYSTEMS[s][0] for s in order], rotation=90)
    bax.set_ylim(0, max(0.05, float(np.nanmax(rngw))) * 1.25)
    bax.text(0.0, 1.0, "(b)", transform=bax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.215, hspace=0.16)
    save(fig, "Fig2")      # measured benchmark: second figure of the paper


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3}[w]()

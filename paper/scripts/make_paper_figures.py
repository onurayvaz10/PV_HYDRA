"""Figures 1-3 of the paper, regenerated from the result files (no values read from images).

Fig. 1  (a) measured locations by data group and the 20 exact-rate weather sites on the 1-km Koppen-Geiger map
        (Beck et al. 2023); (b) structure of kappa-HYDRA-D.                                  181 mm (two columns)
Fig. 2  (a) injected-change error, unit = system x injected rate (T2_recovery_by_system.csv);
        (b) exact-rate absolute error, unit = weather site (E7_semisynthetic_by_site.csv).    88 mm (one column)
Fig. 3  measured long sensor-grade records: (a) reference, seven two-stage and kappa-HYDRA-D rates per system
        (T1_plr_by_system.csv); (b) reference-window range of the XGBoost rate (T12_refwindow_sensitivity.csv).

Okabe-Ito palette, distinct marker shapes (readable in grey scale), 8 pt text at final size, panel labels (a), (b).
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REV = Path(__file__).resolve().parents[1]          # paper/
ROOT = REV.parent
sys.path.insert(0, str(ROOT))
TAB = ROOT / "results/degradation_paper/tables"
EXP = ROOT / "results/climate_expansion"
OUT = REV / "figures"
MM = 1 / 25.4
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
METHODS = ["rdtools", "khd_full", "xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
LABEL = {"rdtools": "Reference", "khd_full": "κ-HYDRA-D", "xgboost": "XGBoost", "random_forest": "Random forest",
         "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "patchtst": "PatchTST", "informer": "Informer"}

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.dpi": 600, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def panel(ax, letter: str, x: float = -0.02, y: float = 1.0) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


# ----------------------------------------------------------------------------------------------- Fig. 1
def group_locations() -> pd.DataFrame:
    """Measured locations with the data group of Table I of the paper."""
    from scripts.make_world_map import measured_locations
    loc = measured_locations()
    sel = pd.read_csv(EXP / "selection.csv")
    sel = sel[(sel.role == "main") & (sel.status == "final")]
    tier = sel.groupby("location").tier.first()
    def grp(r):
        if r.group == "paper":
            return "long sensor-grade"
        if r.group == "short record":
            return "short"
        return "long expansion, on-site POA" if tier.get(r.location) == "S" else "long expansion, ERA5"
    loc["data_group"] = loc.apply(grp, axis=1)
    return loc


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
    loc = group_locations()
    semi = pd.read_csv(EXP / "semisynthetic_sites.csv")

    fig = plt.figure(figsize=(181 * MM, 80 * MM))
    ax = fig.add_axes([0.075, 0.30, 0.55, 0.62])
    ax.imshow(np.ma.masked_equal(data, 0), cmap=cmap, vmin=0, vmax=len(codes) - 1, alpha=0.35,
              extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
    ax.scatter(semi.longitude, semi.latitude, marker="^", s=26, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
    style = {"long sensor-grade": ("o", OI["black"]), "long expansion, on-site POA": ("D", OI["verm"]),
             "long expansion, ERA5": ("s", OI["blue"]), "short": ("v", OI["orange"])}
    for g, (mk, col) in style.items():
        part = loc[loc.data_group == g]
        ax.scatter(part.lon, part.lat, marker=mk, s=22 + 7 * np.sqrt(part.systems), facecolor=col,
                   edgecolor="white", lw=0.6, zorder=4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-50, -25, 0, 25, 50, 75])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.text(-0.13, 1.03, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    n = loc.groupby("data_group").systems.sum()
    handles = [Line2D([], [], marker="o", ls="", mfc=OI["black"], mec="white", ms=6,
                      label=f"Long sensor-grade ({n['long sensor-grade']} systems)"),
               Line2D([], [], marker="D", ls="", mfc=OI["verm"], mec="white", ms=5.5,
                      label=f"Long expansion, on-site POA ({n['long expansion, on-site POA']})"),
               Line2D([], [], marker="s", ls="", mfc=OI["blue"], mec="white", ms=5.5,
                      label=f"Long expansion, ERA5 ({n['long expansion, ERA5']})"),
               Line2D([], [], marker="v", ls="", mfc=OI["orange"], mec="white", ms=6,
                      label=f"Short records, 2–3 yr ({n['short']})"),
               Line2D([], [], marker="^", ls="", mfc="white", mec="black", ms=6,
                      label=f"Exact-rate weather site ({len(semi)})")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.35)

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
    box(d, xs[0], 0.66, w, 0.13, c_net, "Correction\n$g_q(x)$")
    box(d, xs[1], 0.66, w, 0.13, c_phys, "PVWatts\n$P_{PVW}$")
    box(d, xs[2], 0.66, w, 0.13, c_tr, "Trend\n$1+ρ(t-t_m)$")
    for m in mid:
        arrow(d, (m, 0.835), (m, 0.79))
    d.scatter([0.5], [0.605], s=110, marker="o", facecolor="white", edgecolor="#333333", lw=0.7, zorder=5)
    d.text(0.5, 0.604, "×", ha="center", va="center", fontsize=9, zorder=6)
    arrow(d, (mid[0], 0.66), (0.47, 0.612))
    arrow(d, (mid[1], 0.66), (0.5, 0.632))
    arrow(d, (mid[2], 0.66), (0.53, 0.612))
    box(d, 0.01, 0.435, 0.98, 0.12, c_out, "Quantiles q = 0.1, 0.5, 0.9\n$\\hat{P}_q = P_{PVW}\\,g_q(x)\\,[1+ρ(t-t_m)]$")
    arrow(d, (0.5, 0.578), (0.5, 0.555))
    box(d, 0.01, 0.225, 0.98, 0.17, "#fff6cc", "Joint fit on the whole record\ninsolation-weighted pinball loss;\nheld-out weeks for hourly scores")
    arrow(d, (0.5, 0.435), (0.5, 0.395))
    box(d, 0.01, 0.015, 0.98, 0.17, c_out, "Outputs\nrate r and 95 % interval;\nhourly 80 % band")
    arrow(d, (0.5, 0.225), (0.5, 0.185))
    save(fig, "Fig1")


# ----------------------------------------------------------------------------------------------- Fig. 2
def strip_box(ax, values: list[np.ndarray], colors: list[str]) -> None:
    rng = np.random.default_rng(0)
    bp = ax.boxplot(values, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 0.9}, boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6},
                    capprops={"lw": 0.6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for i, v in enumerate(values, 1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=3, color="black", lw=0, alpha=0.6, zorder=3)


def fig2() -> None:
    rec = pd.read_csv(TAB / "T2_recovery_by_system.csv")
    e7 = pd.read_csv(EXP / "tables/E7_semisynthetic_by_site.csv")
    colors = [OI["grey"], OI["verm"], OI["blue"], OI["sky"], OI["green"], OI["green"], OI["orange"], OI["purple"],
              OI["purple"]]
    fig, axes = plt.subplots(2, 1, figsize=(88 * MM, 88 * MM), sharex=True)
    va = [rec[rec.method == m].error.abs().to_numpy() for m in METHODS]
    vb = [e7[e7.method == m].mae.to_numpy() for m in METHODS]
    for ax, vals, ylab in ((axes[0], va, "Injected-change error (%/yr)"), (axes[1], vb, "Exact-rate error (%/yr)")):
        strip_box(ax, vals, colors)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", lw=0.3, color="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(1e-4, 4)
    axes[1].set_ylim(1e-3, 0.5)
    axes[1].set_xticks(range(1, len(METHODS) + 1), [LABEL[m] for m in METHODS], rotation=45, ha="right")
    for ax_, l_ in ((axes[0], "a"), (axes[1], "b")):
        ax_.text(0.015, 0.97, f"({l_})", transform=ax_.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    axes[0].text(0.99, 0.97, "unit: system × rate, n = 60", transform=axes[0].transAxes, ha="right", va="top")
    axes[1].text(0.99, 0.97, "unit: weather site, n = 20", transform=axes[1].transAxes, ha="right", va="top")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.955, bottom=0.19, hspace=0.2)
    save(fig, "Fig2")


# ----------------------------------------------------------------------------------------------- Fig. 3
TECH = {"CdTe": "CdTe", "Amorphous Silicon": "a-Si", "mono-Si": "m-Si", "poly-Si": "p-Si", "HIT Hybrid Silicon": "HIT"}


def short_label(sid: str, label: str) -> str:
    if sid.startswith("dkasc"):
        tech = next((v for k, v in TECH.items() if k in label), "")
        return f"A{sid.split('_')[1]} {tech}"
    return {"pvdaq_1430": "Golden", "pvdaq_1423": "Henderson", "pvdaq_2107": "Arbuckle"}[sid]


def fig3() -> None:
    t1 = pd.read_csv(TAB / "T1_plr_by_system.csv")
    t12 = pd.read_csv(TAB / "T12_refwindow_sensitivity.csv").set_index("system_id")
    ml = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
    two = t1[[f"{m}_plr_mean" for m in ml]]
    x = np.arange(len(t1))
    labels = [short_label(s, l) for s, l in zip(t1.system_id, t1.label)]
    fig, axes = plt.subplots(2, 1, figsize=(88 * MM, 76 * MM), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
    a = axes[0]
    a.vlines(x, two.min(axis=1), two.max(axis=1), color=OI["blue"], lw=3.2, alpha=0.45, zorder=2)
    for m in ml:
        a.scatter(x, t1[f"{m}_plr_mean"], s=6, color=OI["blue"], lw=0, zorder=3)
    a.errorbar(x - 0.28, t1.rdtools_plr, yerr=[t1.rdtools_plr - t1.rdtools_ci_low, t1.rdtools_ci_high - t1.rdtools_plr],
               fmt="o", ms=3, color="black", lw=0.6, capsize=0, zorder=4)
    a.errorbar(x + 0.28, t1.khd_plr, yerr=[t1.khd_plr - t1.khd_ci_low, t1.khd_ci_high - t1.khd_plr],
               fmt="D", ms=3, color=OI["verm"], lw=0.6, capsize=0, zorder=4)
    a.axhline(0, color="#999999", lw=0.5)
    a.set_ylabel("Rate (%/yr)")
    a.grid(axis="y", lw=0.3, color="#dddddd")
    a.spines[["top", "right"]].set_visible(False)
    a.legend(handles=[Line2D([], [], marker="o", color="black", ms=3, lw=0.6, label="Reference, 95 % CI"),
                      Line2D([], [], color=OI["blue"], lw=3.2, alpha=0.45, label="Seven two-stage rates"),
                      Line2D([], [], marker="D", color=OI["verm"], ms=3, lw=0.6, label="κ-HYDRA-D, 95 % CI")],
             loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.1)
    a.text(0.62, 0.97, "(a)", transform=a.transAxes, ha="center", va="top", fontsize=8, fontweight="bold")
    bax = axes[1]
    rngw = t12.loc[t1.system_id, "window_range"].to_numpy()
    bax.bar(x, rngw, width=0.6, color=OI["orange"], edgecolor="black", lw=0.4)
    bax.axhline(np.median(rngw), color="black", lw=0.7, ls="--")
    bax.text(1.5, np.median(rngw) + 0.008, "median", ha="center", va="bottom")
    bax.set_ylabel("Window range (%/yr)")
    bax.spines[["top", "right"]].set_visible(False)
    bax.set_xticks(x, labels, rotation=90)
    bax.set_ylim(0, 0.33)
    bax.text(0.985, 0.97, "(b)", transform=bax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.965, bottom=0.235, hspace=0.12)
    save(fig, "Fig3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()

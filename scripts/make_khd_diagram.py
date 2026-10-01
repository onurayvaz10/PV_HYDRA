"""Architecture diagram of κ-HYDRA-D (manuscript figure), drawn from the implementation in
src/degradation/kappa_hydra_d.py: inputs without time, PVWatts anchor, bounded correction network with non-crossing
quantile heads, explicit linear trend, insolation-weighted pinball loss, rate and interval.
Output: reports/manuscript/figures/F_khd_architecture.png/.pdf (IEEE one-column width).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/manuscript/figures"
C = {"in": "#e8eef7", "phys": "#fde7c8", "net": "#e2f0d9", "trend": "#f3e1f0", "out": "#eeeeee", "loss": "#fff6cc"}


def box(ax, x, y, w, h, color, title, body="", size=6.8, gap=0.038):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=color,
                                ec="#444444", lw=0.6))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=size, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.012 - gap, body, ha="center", va="top", fontsize=size - 0.6, linespacing=1.3)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333",
                                 shrinkA=0, shrinkB=0, **kw))


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "savefig.dpi": 600, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.5, 2.6))
    ax = fig.add_axes([0, -0.048, 1, 1.05])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, xs = 0.315, (0.01, 0.3425, 0.675)
    mid = [x + w / 2 for x in xs]

    box(ax, xs[0], 0.815, w, 0.175, C["in"], "Weather inputs x",
        "POA, $T_{cell}$, $T_{amb}$,\nhour, day of year (sin, cos);\nno time or age input")
    box(ax, xs[1], 0.815, w, 0.175, C["in"], "Plant data", "$P_{rated}$ (catalog),\ndatasheet γ")
    box(ax, xs[2], 0.815, w, 0.175, C["in"], "Time t", "years, centred on\nmid-record $t_{mid}$")

    box(ax, xs[0], 0.515, w, 0.275, C["net"], "Correction $g_q(x)$",
        "MLP 7–128–128–3, GELU\n$g_{0.5} = 1 + 0.5\\,\\tanh z_2$\n(bounded 0.5–1.5)\n$g_{0.1}, g_{0.9} = g_{0.5} \\mp$ softplus\n(non-crossing)")
    box(ax, xs[1], 0.515, w, 0.275, C["phys"], "Physics anchor",
        "PVWatts dc power\n$P_{PVWatts} = P_{rated}\\,\\frac{POA}{1000}$\n$\\times\\,[1 + γ\\,(T_{cell} - 25)]$")
    box(ax, xs[2], 0.515, w, 0.275, C["trend"], "Trend",
        "$1 + ρ\\,(t - t_{mid})$\none scalar ρ per system;\nρ frozen for the first\n30 % of the steps")
    for m in mid:
        arrow(ax, (m, 0.815), (m, 0.79))

    ax.add_patch(plt.Circle((0.50, 0.465), 0.03, fc="white", ec="#333333", lw=0.7))
    ax.text(0.50, 0.463, "×", ha="center", va="center", fontsize=9)
    arrow(ax, (mid[0], 0.515), (0.472, 0.472))
    arrow(ax, (mid[1], 0.515), (0.50, 0.497))
    arrow(ax, (mid[2], 0.515), (0.528, 0.472))
    ax.add_patch(FancyBboxPatch((0.18, 0.305), 0.64, 0.105, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=C["out"], ec="#444444", lw=0.6))
    ax.text(0.50, 0.375, "$\\hat{P}_q = P_{PVWatts}\\; g_q(x)\\; [1 + ρ\\,(t - t_{mid})]$", ha="center", va="center",
            fontsize=7)
    ax.text(0.50, 0.33, "quantiles $q$ = 0.1, 0.5, 0.9", ha="center", va="center", fontsize=5.8)
    arrow(ax, (0.50, 0.435), (0.50, 0.41))

    box(ax, 0.01, 0.06, 0.485, 0.19, C["loss"], "Fitting",
        "filtered hours except held-out weeks;\ninsolation-weighted pinball loss;\nresponse and trend fitted jointly")
    box(ax, 0.505, 0.06, 0.485, 0.19, C["out"], "Outputs",
        "rate $r = ρ\\,/\\,[1 + ρ\\,(0.5 - t_{mid})]$ (%/yr);\n95 % CI: leave-one-year-out\njackknife and seed spread")
    arrow(ax, (0.36, 0.305), (0.27, 0.25))
    arrow(ax, (0.64, 0.305), (0.73, 0.25))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F_khd_architecture.{ext}")
    plt.close(fig)
    print("wrote", FIG / "F_khd_architecture.png")


if __name__ == "__main__":
    main()

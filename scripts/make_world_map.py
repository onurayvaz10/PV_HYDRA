"""World map of the evidence: measured locations (paper and climate expansion) and the 20 known-truth weather sites
(one per country) on the 1-km Koppen-Geiger map for 1991-2020 (Beck et al. 2023, CC BY 4.0), drawn in its published
colours at reduced resolution.

Inputs: results/degradation_paper/tables/T0_sites.csv, results/climate_expansion/selection.csv (main systems),
        results/climate_expansion/screen_tier_*.csv (coordinates), results/climate_expansion/semisynthetic_sites.csv
Output: reports/manuscript/figures/F_world_map.png/.pdf, results/climate_expansion/tables/E10_measured_locations.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_global import country_of  # noqa: E402

CLIM = ROOT / "data/reference/climate"
FIG = ROOT / "reports/manuscript/figures"
OUT = ROOT / "results/climate_expansion"


def legend() -> dict[int, tuple[str, tuple]]:
    out = {}
    for ln in (CLIM / "legend.txt").read_text(encoding="latin-1").splitlines():
        m = re.match(r"\s*(\d+):\s+(\S+)\s+(.+?)\s+\[(\d+) (\d+) (\d+)\]", ln)
        if m:
            out[int(m.group(1))] = (m.group(2), tuple(int(m.group(k)) / 255 for k in (4, 5, 6)))
    return out


def measured_locations() -> pd.DataFrame:
    paper = pd.read_csv(ROOT / "results/degradation_paper/tables/T0_sites.csv")
    paper = paper[paper.retained].groupby("location").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                                          climate=("climate", "first"), systems=("system_id", "size"),
                                                          country=("country", "first")).reset_index()
    paper["group"] = "paper"
    paper["country"] = paper.country.replace({"USA": "United States"})
    sel = pd.read_csv(OUT / "selection.csv")
    sel = sel[(sel.role == "main") & (sel.status == "final")]
    scr = pd.concat([pd.read_csv(OUT / f) for f in ("screen_tier_s.csv", "screen_tier_r.csv")])
    sel = sel.merge(scr[["system_id", "lat", "lon"]].drop_duplicates("system_id"), on="system_id", how="left")
    exp = sel.groupby("location").agg(lat=("lat", "mean"), lon=("lon", "mean"), climate=("climate", "first"),
                                      systems=("system_id", "size"), first=("system_id", "first")).reset_index()
    exp["country"] = exp["first"].map(country_of)
    exp["group"] = np.where(exp["first"].str.startswith(("hk_", "ut_", "fmi_")), "short record", "expansion")
    exp = exp.drop(columns="first")
    return pd.concat([paper, exp], ignore_index=True)


def main() -> None:
    lut = legend()
    with rasterio.open(CLIM / "koppen_geiger_1991_2020_0p00833333.tif") as src:
        data = src.read(1, out_shape=(1080, 2160), resampling=Resampling.mode)
        b = src.bounds
    codes = np.arange(0, max(lut) + 1)
    colours = [(1, 1, 1)] + [lut.get(c, ("", (1, 1, 1)))[1] for c in codes[1:]]
    cmap = ListedColormap(colours)
    loc = measured_locations()
    loc.to_csv(OUT / "tables/E10_measured_locations.csv", index=False)
    semi = pd.read_csv(OUT / "semisynthetic_sites.csv")

    fig, ax = plt.subplots(figsize=(7.16, 3.6), dpi=300)
    ax.imshow(np.ma.masked_equal(data, 0), cmap=cmap, vmin=0, vmax=len(colours) - 1, alpha=0.55,
              extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
    ax.scatter(semi.longitude, semi.latitude, marker="^", s=30, facecolor="white", edgecolor="black", linewidth=0.8,
               zorder=3)
    styles = {"paper": ("o", "#1b1b1b"), "expansion": ("o", "#d62728"), "short record": ("s", "#1f77b4")}
    for g, (mk, col) in styles.items():
        part = loc[loc.group == g]
        ax.scatter(part.lon, part.lat, marker=mk, s=18 + 6 * np.sqrt(part.systems), facecolor=col, edgecolor="white",
                   linewidth=0.6, zorder=4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks(range(-150, 181, 50))
    ax.set_yticks(range(-50, 76, 25))
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Longitude (°)", fontsize=7)
    ax.set_ylabel("Latitude (°)", fontsize=7)
    handles = [Line2D([], [], marker="o", ls="", mfc="#1b1b1b", mec="white", ms=6, label="Measured, original study (2 countries)"),
               Line2D([], [], marker="o", ls="", mfc="#d62728", mec="white", ms=6, label="Measured, climate expansion"),
               Line2D([], [], marker="s", ls="", mfc="#1f77b4", mec="white", ms=6, label="Measured, short record (2–3 y)"),
               Line2D([], [], marker="^", ls="", mfc="white", mec="black", ms=6, label="Known-truth weather site (one per country)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6, frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "F_world_map.png", dpi=300)
    fig.savefig(FIG / "F_world_map.pdf")
    print(loc.to_string(index=False))


if __name__ == "__main__":
    main()

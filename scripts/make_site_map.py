"""Site map and climate of the four locations of the retained systems (manuscript Fig. 1, Table II).

Map: the 1-km Köppen–Geiger raster for 1991–2020 (Beck et al. 2023, CC BY 4.0; data/reference/climate), drawn in its
published colours, with the site coordinates of results/degradation_paper/tables/T0_sites.csv (make_site_table.py).
Climate: annual global horizontal irradiation from ERA5 (Open-Meteo archive) at the same coordinates for the common
decade 2014–2023, and the on-site measured air temperature over the analyzed record (mean, and mean daily maximum on
days with at least 20 logged hours). ERA5 air temperature is kept in the table for comparison only: at Arbuckle its
0.25° cell is about 3 °C warmer than the on-site sensor.
Outputs: tables/T0b_site_climate.csv, reports/manuscript/figures/F0_site_map.png/.pdf
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
from matplotlib.patches import Patch  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation.tier_a import load_system  # noqa: E402
from src.weather.open_meteo import reanalysis  # noqa: E402

TAB = ROOT / "results/degradation_paper/tables"
FIG = ROOT / "reports/manuscript/figures"
CLIM = ROOT / "data/reference/climate"
YEARS = ("2014-01-01", "2023-12-31")
SHORT = {"Alice Springs, NT": "Alice Springs", "Henderson, NV": "Henderson", "Golden, CO": "Golden",
         "Arbuckle, CA": "Arbuckle"}
PANELS = {"us": (-124.5, -101.5, 33.0, 42.5), "au": (113.0, 154.0, -39.5, -10.5)}   # lon0, lon1, lat0, lat1


def legend() -> dict[int, tuple[str, str, tuple]]:
    out = {}
    for ln in (CLIM / "legend.txt").read_text(encoding="latin-1").splitlines():
        m = re.match(r"\s*(\d+):\s+(\S+)\s+(.+?)\s+\[(\d+) (\d+) (\d+)\]", ln)
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3).strip(), tuple(int(m.group(k)) / 255 for k in (4, 5, 6)))
    return out


def site_climate(sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in sites.iterrows():
        w, meta = reanalysis(s.latitude, s.longitude, *YEARS, ROOT / "data/cache/weather",
                             variables=("shortwave_radiation", "temperature_2m"))
        w = w.loc[YEARS[0]:YEARS[1] + " 23:00"]
        n_years = w.index.year.nunique()
        solar = w.index + pd.Timedelta(hours=s.longitude / 15)        # local solar time for the daily maximum
        tmax = w.temperature_2m.groupby(solar.date).max()
        sid = s.first_system.replace("pvdaq_", "")
        m, _ = load_system(sid)                                    # measured hourly record (local standard time)
        t = m.t_amb_c.dropna()
        day = t.groupby(t.index.date)
        dmax = day.max()[day.size() >= 20]
        rows.append({"location": s.location, "country": s.country, "latitude": s.latitude, "longitude": s.longitude,
                     "elevation_m": s.elevation_m, "climate": s.climate,
                     "ghi_kwh_m2_yr": w.shortwave_radiation.sum() / 1000 / n_years,
                     "t_mean_c": t.mean(), "t_daily_max_mean_c": dmax.mean(), "t_source_system": s.first_system,
                     "t_hours": len(t), "era5_t_mean_c": w.temperature_2m.mean(), "era5_t_daily_max_mean_c": tmax.mean(),
                     "years": n_years, "hours": int(w.shortwave_radiation.notna().sum()),
                     "era5_grid_lat": meta["grid_latitude"], "era5_grid_lon": meta["grid_longitude"]})
    return pd.DataFrame(rows)


def draw_panel(ax, bounds, lut, sites, ticks):
    lon0, lon1, lat0, lat1 = bounds
    with rasterio.open(CLIM / "koppen_geiger_1991_2020_0p00833333.tif") as src:
        win = from_bounds(lon0, lat0, lon1, lat1, src.transform)
        step = 2                                                   # ~2 km cells are ample at print size
        arr = src.read(1, window=win, out_shape=(int(win.height // step), int(win.width // step)))
    ax.imshow(arr, cmap=lut, vmin=0, vmax=30, extent=(lon0, lon1, lat0, lat1), interpolation="nearest",
              aspect=1 / np.cos(np.deg2rad((lat0 + lat1) / 2)))
    for _, s in sites.iterrows():
        ax.plot(s.longitude, s.latitude, marker="o", ms=5, mfc="white", mec="black", mew=1.0, zorder=5)
        dx, dy, ha = s.label_offset
        ax.annotate(f"{SHORT[s.location]} ({s.climate})", (s.longitude, s.latitude), (s.longitude + dx, s.latitude + dy),
                    fontsize=7, ha=ha, va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xticks(ticks[0], [f"{abs(v):.0f}°{'W' if v < 0 else 'E'}" for v in ticks[0]])
    ax.set_yticks(ticks[1], [f"{abs(v):.0f}°{'S' if v < 0 else 'N'}" for v in ticks[1]])
    ax.tick_params(length=2, pad=1)
    return arr


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = pd.read_csv(TAB / "T0_sites.csv")
    sites = (t0[t0.retained].groupby("location", sort=False)
             .agg(country=("country", "first"), latitude=("latitude", "first"), longitude=("longitude", "first"),
                  elevation_m=("elevation_m", "first"), climate=("climate", "first"), n=("system_id", "size"),
                  first_system=("system_id", "first"))
             .reset_index())
    clim = site_climate(sites)
    clim["n_systems"] = clim.location.map(sites.set_index("location").n)
    clim.to_csv(TAB / "T0b_site_climate.csv", index=False)
    print(clim.round(2).to_string(index=False))

    lg = legend()
    colors = [(1, 1, 1)] + [tuple(0.25 + 0.75 * c for c in lg[k][2]) if k in lg else (1, 1, 1) for k in range(1, 31)]
    lut = ListedColormap(colors)                                   # Beck et al. colours, lightened for the markers
    offsets = {"Alice Springs, NT": (0.0, -8.5, "center"), "Henderson, NV": (-1.0, -2.6, "center"),
               "Golden, CO": (0.3, 1.8, "center"), "Arbuckle, CA": (0.6, 2.0, "left")}
    sites["label_offset"] = sites.location.map(offsets)

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "savefig.dpi": 600, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.5, 3.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 0.95], hspace=0.32, wspace=0.42,
                          left=0.1, right=0.9, top=0.97, bottom=0.23)
    a = fig.add_subplot(gs[0, :])
    us = draw_panel(a, PANELS["us"], lut, sites[sites.country == "USA"], ([-120, -115, -110, -105], [34, 38, 42]))
    a.set_title("(a) Western United States", fontsize=7, loc="left", pad=2)
    b = fig.add_subplot(gs[1, 0])
    au = draw_panel(b, PANELS["au"], lut, sites[sites.country == "Australia"], ([120, 135, 150], [-35, -25, -15]))
    b.set_title("(b) Australia", fontsize=7, loc="left", pad=2)

    c = fig.add_subplot(gs[1, 1])
    order = clim.sort_values("ghi_kwh_m2_yr", ascending=False).reset_index(drop=True)
    x = np.arange(len(order))
    c.bar(x, order.ghi_kwh_m2_yr, color="#9fb7d9", edgecolor="black", lw=0.4, width=0.62)
    c.set_ylabel("ERA5 GHI (kWh m$^{-2}$ yr$^{-1}$)", fontsize=6.5)
    c.set_ylim(0, 2600)
    c.set_xticks(x, [f"{SHORT[v]}" for v in order.location], rotation=35, ha="right", fontsize=6)
    c2 = c.twinx()
    c2.plot(x, order.t_mean_c, "o", color="#d55e00", ms=3.5, label="mean")
    c2.plot(x, order.t_daily_max_mean_c, "^", color="#7a1f00", ms=3.5, label="daily max.")
    c2.set_ylim(0, 35)
    c2.set_ylabel("On-site air temperature (°C)", fontsize=6.5, color="#7a1f00")
    c2.tick_params(labelsize=6)
    c2.legend(fontsize=5.5, loc="upper right", ncol=2, frameon=False, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.2)
    c.set_title("(c) Site climate", fontsize=7, loc="left", pad=2)
    c.tick_params(length=2, pad=1)

    keep = {k for k, v in lg.items() if v[0] in set(sites.climate)}
    for arr in (us, au):                                           # every class covering >= 2 % of a panel's land
        share = pd.Series(arr.ravel())
        share = share[share > 0].value_counts(normalize=True)
        keep |= set(share[share >= 0.02].index)
    keep = sorted(keep)
    handles = [Patch(fc=colors[k], ec="black", lw=0.3, label=lg[k][0]) for k in keep]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=6, frameon=False,
               handlelength=1.0, handleheight=0.8, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.5, 0.0),
               title="Köppen–Geiger class (1991–2020)", title_fontsize=6)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F0_site_map.{ext}")
    plt.close(fig)
    print("classes in legend:", [lg[k][0] for k in keep])


if __name__ == "__main__":
    main()

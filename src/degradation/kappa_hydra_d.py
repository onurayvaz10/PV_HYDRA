"""kappa-HYDRA-D: physics-anchored, single-stage PLR estimation with quantile heads.

    P_hat_q(x, t) = P_PVWatts(POA, T_cell) * g_q(x) * (1 + rho * (t - t_mid))

  P_PVWatts   RdTools PVWatts DC power (rated power, POA, gamma, cell temperature): the physics term.
  g_q(x)      bounded correction for quantile q in {0.1, 0.5, 0.9}: median = 1 + 0.5 tanh(.), lower/upper
              = median -/+ softplus(.) (non-crossing). Captures angle-of-incidence, spectral, low-light and
              inverter effects. Its inputs contain NO time or age variable (only POA, T_cell, T_amb and
              periodic sun/season terms). Explicit linear time dependence is carried by rho.
              Weather can proxy time, so excluding age does not prove identifiability.
  rho         one scalar per system, the rate relative to mid-record performance (centred time); it is
              reported relative to the first-year level, r = rho / (1 + rho (0.5 - t_mid)), the convention
              of first-year-normalised YoY. PLR = 100 r (%/yr).

Fitted on all hours that pass the RdTools filters and the outage-day screen (target: kappa = P / P_PVWatts),
insolation-weighted pinball loss; rho is frozen for the first 30 % of the steps (g learns the level first).
Uncertainty of r: leave-one-year-out jackknife (seed 0) combined with the spread over seeds. Held-out
blocked weeks (every 5th ISO week) give RMSE/MAE/R2 in P/P_rated units and quantile coverage.

Ablation variants: no_physics (target P/P_rated, no PVWatts term), no_correction (constant g),
time_input (diagnostic: g also sees time, which breaks identifiability on purpose).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import rdtools
import torch
from torch import nn

from src.degradation.ml_plr import clean, healthy_hours
from src.degradation.rdtools_pipeline import cell_temperature, standard_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU fallback (e.g. a Colab runtime without GPU)

QUANTILES = (0.1, 0.5, 0.9)
FEATURES = ["poa_n", "tcell_n", "tamb_n", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
VARIANTS = {"full": {}, "no_physics": {"physics": False}, "no_correction": {"correction": False},
            "time_input": {"time_input": True},   # time_input: diagnostic, breaks identifiability on purpose
            "mean_loss": {"loss": "mse"}}           # median head fitted to the insolation-weighted mean (diagnostic)


class KappaHydraD(nn.Module):
    def __init__(self, features: int, hidden: int = 64, dropout: float = 0.05, fixed_rate: float | None = None,
                 physics: bool = True) -> None:
        super().__init__()
        self.physics = physics
        self.body = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(QUANTILES)))
        self.rate = nn.Parameter(torch.zeros(()), requires_grad=fixed_rate is None)
        if fixed_rate is not None:
            self.rate.data.fill_(fixed_rate)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        # With the physics term g is a bounded correction; without it g must carry the whole response.
        median = 1.0 + 0.5 * torch.tanh(z[:, 1]) if self.physics else nn.functional.softplus(z[:, 1])
        # Non-crossing quantiles: q10 = median - softplus, q90 = median + softplus.
        g = torch.stack([median - nn.functional.softplus(z[:, 0]), median, median + nn.functional.softplus(z[:, 2])], 1)
        return g * (1.0 + self.rate * t_years).unsqueeze(1)


def design(frame: pd.DataFrame, dc_kw: float, gamma: float, physics: bool = True) -> pd.DataFrame:
    frame = clean(frame)
    poa = frame.poa_wm2.clip(lower=0)
    tcell = cell_temperature(poa, frame.get("t_amb_c"), frame.get("t_module_c"))
    expected = rdtools.normalization.pvwatts_dc_power(poa, dc_kw * 1000.0, temperature_cell=tcell, gamma_pdc=gamma)
    power = frame.power_w.clip(lower=0)
    kappa = power / expected.replace(0, np.nan)
    mask = standard_mask(kappa, poa, tcell, power) & healthy_hours(frame, dc_kw)
    idx = frame.index
    hour = idx.hour + 0.5
    tamb = frame.t_amb_c if "t_amb_c" in frame else tcell
    out = pd.DataFrame({"poa_n": poa / 1000.0, "tcell_n": tcell / 60.0, "tamb_n": tamb.fillna(tcell) / 40.0,
                        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                        "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25),
                        "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25),
                        "kappa": kappa if physics else power / (dc_kw * 1000.0), "poa": poa,
                        # kappa * scale = P / P_rated: metrics in the same unit as the two-stage models
                        "scale": expected / (dc_kw * 1000.0) if physics else pd.Series(1.0, index=idx),
                        "t_years": (idx - idx.min()).total_seconds() / (365.25 * 86400)}, index=idx)
    out = out[mask.reindex(idx, fill_value=False) & out.notna().all(axis=1)]
    # Centred time: level (g) and slope (rho) decouple; rho is the rate relative to mid-record performance.
    out["span"] = out.t_years.max() - out.t_years.min()
    out["t_mid"] = (out.t_years.max() + out.t_years.min()) / 2
    out["t_mean"] = out.t_years.mean()
    out["t_years"] = out.t_years - out.t_mid
    out["t_input"] = out.t_years / 10.0
    out["val"] = out.index.isocalendar().week.to_numpy() % 5 == 0
    return out


def _inputs(data: pd.DataFrame, variant: dict, device: str) -> torch.Tensor:
    cols = FEATURES + (["t_input"] if variant.get("time_input") else [])
    x = torch.as_tensor(data[cols].to_numpy(np.float32), device=device)
    return x if variant.get("correction", True) else torch.zeros_like(x)


def _fit(data: pd.DataFrame, seed: int, hidden: int, lr: float, steps: int, device: str,
         fixed_rate: float | None = None, variant: dict | None = None) -> tuple[KappaHydraD, float]:
    variant = variant or {}
    physics = variant.get("physics", True)
    torch.manual_seed(seed)
    train = data[~data.val]
    x = _inputs(train, variant, device)
    t = torch.as_tensor(train.t_years.to_numpy(np.float32), device=device)
    y = torch.as_tensor(train.kappa.to_numpy(np.float32), device=device)
    w = torch.as_tensor((train.poa / train.poa.mean()).to_numpy(np.float32), device=device)
    q = torch.as_tensor(QUANTILES, device=device)
    net = KappaHydraD(x.shape[1], hidden, fixed_rate=fixed_rate, physics=physics).to(device)
    opt = torch.optim.Adam([{"params": net.body.parameters(), "lr": lr, "weight_decay": 1e-5},
                            {"params": [net.rate], "lr": lr * 0.1}])
    with torch.no_grad():                                   # start g at the median kappa level
        level = y.median()
        start = torch.atanh(torch.clamp((level - 1) / 0.5, -0.99, 0.99)) if physics else torch.log(torch.expm1(level.clamp(min=1e-3)))
        net.body[-1].bias.copy_(torch.tensor([-3.0, float(start), -3.0], device=device))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = min(len(train), 8192)
    warmup = int(0.3 * steps)          # g learns level and shape first; rho is released afterwards
    for step in range(steps):
        net.rate.requires_grad_(fixed_rate is None and step >= warmup)
        i = torch.randint(len(train), (batch,), generator=gen).to(device)
        err = y[i].unsqueeze(1) - net(x[i], t[i])
        pin = torch.maximum(q * err, (q - 1) * err)
        if variant.get("loss") == "mse":             # central head: weighted squared error (a mean, not a median)
            pin = torch.cat([pin[:, :1], 20.0 * err[:, 1:2] ** 2, pin[:, 2:]], dim=1)
        loss = (pin * w[i].unsqueeze(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net.eval(), float(net.rate.detach().cpu())


def _predict(net: KappaHydraD, data: pd.DataFrame, device: str, variant: dict | None = None) -> np.ndarray:
    with torch.no_grad():
        return net(_inputs(data, variant or {}, device),
                   torch.as_tensor(data.t_years.to_numpy(np.float32), device=device)).cpu().numpy()


def estimate(frame: pd.DataFrame, dc_kw: float, gamma: float, seeds=(0, 1, 2, 3, 4), hidden: int = 64,
             lr: float = 3e-3, steps: int = 3000, jackknife: bool = True, device: str = DEVICE,
             variant: str = "full") -> dict:
    started = time.perf_counter()
    spec = VARIANTS[variant]
    data = design(frame, dc_kw, gamma, physics=spec.get("physics", True))
    if len(data) < 2000 or data.span.iloc[0] < 2:
        return {"status": "insufficient data", "rows": len(data)}
    rates, preds = [], []
    for seed in seeds:
        net, rate = _fit(data, seed, hidden, lr, steps, device, variant=spec)
        rates.append(rate)
        preds.append(_predict(net, data[data.val], device, spec))
    pred = np.mean(preds, axis=0)
    held = data[data.val]
    scale = held.scale.to_numpy()[:, None]
    pred = pred * scale
    y = held.kappa.to_numpy() * scale[:, 0]
    err = pred[:, 1] - y
    # rho is estimated relative to mid-record performance; it is reported relative to the first-year level
    # (t = 0.5), the convention of first-year-normalised YoY, so all methods share one definition.
    t_mid = float(data.t_mid.iloc[0])
    to_first = lambda rho: rho / (1.0 + rho * (0.5 - t_mid))  # noqa: E731
    rates_mid = rates
    rates = [to_first(x) for x in rates_mid]
    result = {"status": "ok", "variant": variant, "plr": 100 * float(np.mean(rates)), "plr_seed_sd": 100 * float(np.std(rates, ddof=1)),
              "plr_mid_reference": 100 * float(np.mean(rates_mid)), "t_center_years": float(data.t_mean.iloc[0]),
              "rows": int(len(data)), "span_years": float(data.span.iloc[0]),
              "val_rmse": float(np.sqrt(np.mean(err ** 2))), "val_mae": float(np.mean(np.abs(err))),
              "val_r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
              "coverage_80": float(np.mean((y >= pred[:, 0]) & (y <= pred[:, 2]))),
              "interval_width_80": float(np.mean(pred[:, 2] - pred[:, 0]))}
    if jackknife:
        years = data.index.year
        uniq = np.unique(years)
        jack = np.array([to_first(_fit(data[years != yr], 0, hidden, lr, steps, device, variant=spec)[1]) for yr in uniq])
        n = len(uniq)
        se_jack = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
        se = float(np.sqrt(se_jack ** 2 + (np.std(rates, ddof=1) if len(rates) > 1 else 0) ** 2))
        result.update({"plr_se": 100 * se, "ci_low": result["plr"] - 196 * se, "ci_high": result["plr"] + 196 * se,
                       "jackknife_years": int(n)})
    result["fit_seconds"] = time.perf_counter() - started
    # Held-out predictions (P/P_rated) for common-domain scoring and DM tests against other models.
    result["_val"] = {"stamps": held.index.to_numpy().astype("datetime64[us]").astype("int64"), "y": y,
                      "pred": pred[:, 1], "q10": pred[:, 0], "q90": pred[:, 2]}
    return result

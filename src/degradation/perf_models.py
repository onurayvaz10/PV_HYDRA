"""Performance models for two-stage ML PLR estimation (degradation paper, RQ1).

A performance model f maps weather and geometry to capacity-normalised AC power.
It is trained on the reference period (first 24 months) only; applied forward it
gives a reference-period expected-power response, and PI = P / f(x).
The reference period need not represent an un-degraded or newly installed system.

Models (user guideline list): XGBoost, RandomForest, LSTM, GRU, TCN, PatchTST,
Informer. Sequence models see a 24-h window of inputs ending at the target hour
(seq-to-one); tree models use only the last 3 hours of that window, flattened.
Informer follows Zhou et al. (2021, AAAI, arXiv:2012.07436): ProbSparse
self-attention (top-u queries by the max-mean sparsity measure) with a
convolutional distilling layer between encoder layers.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.baselines import CausalResidualBlock, PatchTST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU fallback (e.g. a Colab runtime without GPU)

FEATURES = ["poa_n", "tcell_n", "tamb_n", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
WINDOW = 24


def feature_frame(frame: pd.DataFrame, gamma: float) -> pd.DataFrame:
    from src.degradation.rdtools_pipeline import cell_temperature
    poa = frame.poa_wm2.clip(lower=0)
    tamb = frame.t_amb_c if "t_amb_c" in frame else pd.Series(np.nan, index=frame.index)
    tcell = cell_temperature(poa, tamb, frame.get("t_module_c"))
    idx = frame.index
    hour = idx.hour + 0.5
    out = pd.DataFrame({"poa_n": poa / 1000.0, "tcell_n": tcell / 60.0, "tamb_n": tamb.fillna(tcell) / 40.0,
                        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                        "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25),
                        "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25)}, index=idx)
    return out


def windows(features: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Hourly grid; a sample exists where all window inputs are present. The target may be
    NaN (such samples are only used for prediction, never for fitting or scoring)."""
    grid = pd.date_range(features.index.min(), features.index.max(), freq="h")
    f = features.reindex(grid).to_numpy(np.float32)
    y = target.reindex(grid).to_numpy(np.float32)
    idx = np.arange(WINDOW - 1, len(grid))
    win = idx[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    x = f[win]
    ok = np.isfinite(x).all(axis=(1, 2)) & (f[idx, 0] > 0)      # daylight target hour
    return x[ok], y[idx][ok], grid[idx][ok]


# ------------------------------------------------------------------ networks
class RecurrentReg(nn.Module):
    def __init__(self, cell: str, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](features, hidden, num_layers=2, batch_first=True,
                                                          dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNReg(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.entry = nn.Conv1d(features, hidden, 1)
        self.blocks = nn.Sequential(*[CausalResidualBlock(hidden, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.entry(x.transpose(1, 2)))[:, :, -1]).squeeze(-1)


class PatchTSTReg(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = PatchTST(features, WINDOW, 1, hidden=hidden, layers=2, heads=4, patch_length=6,
                            patch_stride=3, dropout=dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ProbSparseAttention(nn.Module):
    """Informer ProbSparse self-attention: only the top-u 'active' queries attend;
    the remaining queries receive the mean of V (lazy queries)."""

    def __init__(self, d_model: int, heads: int, factor: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.h, self.dk, self.factor = heads, d_model // heads, factor
        self.q, self.k, self.v, self.o = (nn.Linear(d_model, d_model) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, length, _ = x.shape
        q = self.q(x).view(b, length, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(b, length, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(b, length, self.h, self.dk).transpose(1, 2)
        u = min(length, max(1, int(self.factor * math.ceil(math.log(length + 1)))))
        sample = torch.randint(length, (u,), device=x.device) if self.training else torch.arange(
            0, length, max(1, length // u), device=x.device)[:u]
        scores_sample = q @ k[:, :, sample].transpose(-2, -1) / math.sqrt(self.dk)
        sparsity = scores_sample.max(-1).values - scores_sample.mean(-1)
        top = sparsity.topk(u, dim=-1).indices                                   # [b, h, u]
        out = v.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()     # lazy queries
        q_top = torch.gather(q, 2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk))
        attn = torch.softmax(q_top @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        out.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk), self.drop(attn) @ v)
        return self.o(out.transpose(1, 2).reshape(b, length, -1))


class InformerReg(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(features, hidden)
        self.position = nn.Parameter(torch.zeros(1, WINDOW, hidden))
        self.attn = nn.ModuleList(ProbSparseAttention(hidden, 4, dropout=dropout) for _ in range(layers))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
                                for _ in range(layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        # Distilling: Conv1d + ELU + MaxPool halves the sequence between layers.
        self.distil = nn.ModuleList(nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden),
                                                  nn.ELU(), nn.MaxPool1d(3, stride=2, padding=1))
                                    for _ in range(layers - 1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.embed(x) + self.position[:, : x.shape[1]]
        for i in range(len(self.attn)):
            h = self.n1[i](h + self.attn[i](h))
            h = self.n2[i](h + self.ff[i](h))
            if i < len(self.distil):
                h = self.distil[i](h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1]).squeeze(-1)


NETWORKS = {"lstm": lambda f, h, d: RecurrentReg("lstm", f, h, d), "gru": lambda f, h, d: RecurrentReg("gru", f, h, d),
            "tcn": TCNReg, "patchtst": PatchTSTReg, "informer": InformerReg}
TREES = ("xgboost", "random_forest", "xgboost_w24")
TREE_HOURS = {"xgboost": 3, "random_forest": 3, "xgboost_w24": 24}   # xgboost_w24: same 24-h input as the networks
ALL_MODELS = ("xgboost", "random_forest") + tuple(NETWORKS)   # the seven benchmarked models (xgboost_w24 is a control)


def default_params(model: str) -> dict:
    if model == "xgboost":
        return {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
    if model == "random_forest":
        return {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 5}
    return {"hidden": 64, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 512}


def _flat(x: np.ndarray, model: str = "xgboost") -> np.ndarray:
    return x[:, -TREE_HOURS[model]:, :].reshape(len(x), -1)


def _train_net(net: nn.Module, model: str, params: dict, seed: int, train: tuple, val: tuple, device: str,
               max_epochs: int, patience: int, lr_scale: float = 1.0) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=float(params["learning_rate"]) * lr_scale, weight_decay=1e-4)
    xt, yt = torch.as_tensor(train[0], device=device), torch.as_tensor(train[1], device=device)
    xv, yv = torch.as_tensor(val[0], device=device), torch.as_tensor(val[1], device=device)
    gen = torch.Generator().manual_seed(seed)
    batch = int(params["batch_size"])
    best, best_state, stale = float("inf"), None, 0
    with torch.backends.cudnn.flags(enabled=model not in ("lstm", "gru")):
        for _ in range(max_epochs):
            net.train()
            order = torch.randperm(len(xt), generator=gen).to(device)
            for s in range(0, len(xt), batch):
                i = order[s:s + batch]
                loss = nn.functional.mse_loss(net(xt[i]), yt[i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                score = float(nn.functional.mse_loss(net(xv), yv))
            if score < best - 1e-7:
                best, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    net.load_state_dict(best_state)
    return net.eval()


def fit_model(model: str, params: dict, seed: int, train: tuple, val: tuple, device: str = DEVICE,
              max_epochs: int = 60, patience: int = 8):
    """Fit on train, early-stop on val (validation weeks only). Returns the fitted estimator."""
    if model in ("xgboost", "xgboost_w24"):
        import xgboost as xgb
        est = xgb.XGBRegressor(random_state=seed, tree_method="hist", early_stopping_rounds=50, **params)
        return est.fit(_flat(train[0], model), train[1], eval_set=[(_flat(val[0], model), val[1])], verbose=False)
    if model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params).fit(_flat(train[0]), train[1])
    torch.manual_seed(seed)
    net = NETWORKS[model](train[0].shape[2], int(params["hidden"]), float(params["dropout"])).to(device)
    return _train_net(net, model, params, seed, train, val, device, max_epochs, patience)


def finetune(model: str, pretrained, params: dict, seed: int, train: tuple, val: tuple, device: str = DEVICE):
    """Adapt a pre-trained estimator to local data: extra boosting rounds on top of the pre-trained
    trees (XGBoost), or all weights at 1/10 of the learning rate with early stopping (networks)."""
    import copy
    if model == "xgboost":
        import xgboost as xgb
        extra = dict(params, n_estimators=300)
        est = xgb.XGBRegressor(random_state=seed, tree_method="hist", early_stopping_rounds=30, **extra)
        return est.fit(_flat(train[0]), train[1], eval_set=[(_flat(val[0]), val[1])], verbose=False,
                       xgb_model=copy.deepcopy(pretrained).get_booster())   # never alter a cached pre-trained model
    if model == "random_forest":
        raise ValueError("random forests cannot be fine-tuned")
    net = copy.deepcopy(pretrained)
    return _train_net(net, model, params, seed, train, val, device, 20, 5, lr_scale=0.1)


def predict(model: str, est, x: np.ndarray, device: str = DEVICE) -> np.ndarray:
    if model in TREES:
        return np.asarray(est.predict(_flat(x, model)), dtype=np.float32)
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=model not in ("lstm", "gru")):
        return torch.cat([est(torch.as_tensor(x[s:s + 8192], device=device)).float().cpu()
                          for s in range(0, len(x), 8192)]).numpy()


def n_parameters(model: str, est) -> int:
    if model in ("xgboost", "xgboost_w24"):
        return int(est.get_booster().num_boosted_rounds())
    if model == "random_forest":
        return int(sum(t.tree_.node_count for t in est.estimators_))
    return int(sum(p.numel() for p in est.parameters()))


def fit_predict(model: str, params: dict, seed: int, train: tuple, val: tuple, apply_x: np.ndarray,
                device: str = DEVICE, max_epochs: int = 60, patience: int = 8) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    est = fit_model(model, params, seed, train, val, device, max_epochs, patience)
    fit_seconds = time.perf_counter() - started
    t0 = time.perf_counter()
    pred = predict(model, est, apply_x, device)
    return pred, {"parameters": n_parameters(model, est), "train_seconds": fit_seconds,
                  "inference_seconds": time.perf_counter() - t0, "inference_samples": int(len(apply_x))}

"""
Per-market logistic recalibration:  logit(p') = a + b * logit(p).

b < 1 pulls overconfident probabilities toward the middle. Fit on one season's
contested rows, judged on a later season. Coefficients live in calibration.json;
refit at season boundaries (python -m gambit.calibrate <backtest.parquet>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CAL_PATH = Path(__file__).with_name("calibration.json")


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(df: pd.DataFrame, col: str = "p_model", contested_only: bool = False) -> dict:
    """Default fits on ALL graded thresholds, because the app applies the map to every
    threshold (ladders included); fitting on contested rows only distorted the tails."""
    out = {}
    src = df[df["contested"]] if contested_only else df
    for market, d in src.groupby("market"):
        x, y = _logit(d[col].to_numpy()), d["y"].to_numpy()

        def nll(th):
            z = th[0] + th[1] * x
            return float(np.sum(np.logaddexp(0, z) - y * z))

        res = minimize(nll, x0=[0.0, 1.0], method="L-BFGS-B")
        out[market] = dict(a=round(float(res.x[0]), 4), b=round(float(res.x[1]), 4), n=int(len(d)))
    return out


def load() -> dict:
    return json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}


def apply(p, market: str, cal: dict | None = None):
    cal = load() if cal is None else cal
    c = cal.get(market)
    if not c:
        return p
    z = c["a"] + c["b"] * _logit(np.asarray(p, float))
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1])
    cal = fit(df)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))

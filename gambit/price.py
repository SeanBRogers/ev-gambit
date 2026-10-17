"""
Pricing against DraftKings Predictions contracts.

A contract pays $1. Buying YES at price q (dollars) with fee f(q) has
EV = p - q - f(q). The max buy price is the highest whole-cent q with
p - q - f(q) >= margin.

Fee schedule per contract (SI, verified Oct 15, 2026; DK may change it):
    $0.01-$0.19 -> $0.01   |   $0.20-$0.96 -> $0.02   |   $0.97-$0.99 -> $0.01
"""

from __future__ import annotations

import numpy as np

FEE_TABLE = [(0.01, 0.19, 0.01), (0.20, 0.96, 0.02), (0.97, 0.99, 0.01)]


def dk_fee(price: float) -> float:
    c = round(price, 2)
    for lo, hi, fee in FEE_TABLE:
        if lo <= c <= hi:
            return fee
    raise ValueError(f"price {price} outside $0.01-$0.99")


def prob_at_least(samples: np.ndarray, threshold: float) -> float:
    """P(stat >= threshold). For a book-style line like 62.5, pass 62.5:
    stat >= 62.5 is identical to 'over 62.5'. For a contract '63+', pass 63."""
    return float(np.mean(np.asarray(samples) >= threshold))


def mc_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def max_buy_price(p: float, margin: float = 0.0) -> float | None:
    """Highest price (dollars, whole cents) with p - q - fee(q) >= margin. None if no price works."""
    best = None
    for cents in range(1, 100):
        q = cents / 100
        if p - q - dk_fee(q) >= margin - 1e-12:
            best = q
    return best


def ev_per_contract(p: float, price: float) -> float:
    return p - price - dk_fee(price)


def american_to_prob(odds: float) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)

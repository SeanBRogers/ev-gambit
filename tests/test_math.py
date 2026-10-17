"""Unit tests for the arithmetic that money depends on."""

import math

import numpy as np
import pytest

from gambit import price as PR
from gambit.data import implied_points
from gambit.model import _bootstrap_sums, _eb_strength


def test_fee_schedule_edges():
    assert PR.dk_fee(0.01) == 0.01
    assert PR.dk_fee(0.19) == 0.01
    assert PR.dk_fee(0.20) == 0.02
    assert PR.dk_fee(0.96) == 0.02
    assert PR.dk_fee(0.97) == 0.01
    assert PR.dk_fee(0.99) == 0.01
    with pytest.raises(ValueError):
        PR.dk_fee(1.00)


def test_max_buy_price_respects_fee_and_margin():
    # p = 0.47: at 45c the fee is 2c -> EV 0.00; at 46c EV -0.01
    assert PR.max_buy_price(0.47) == 0.45
    # p = 0.18 with 1c fee below 20c: 17c -> EV 0.00
    assert PR.max_buy_price(0.18) == 0.17
    # a 2c safety margin lowers it
    assert PR.max_buy_price(0.47, margin=0.02) == 0.43
    # nothing works for a tiny probability
    assert PR.max_buy_price(0.01) is None


def test_fee_cliff_at_20c():
    # p = 0.215: 19c (fee 1c) -> EV +0.015; 20c (fee 2c) -> -0.005. Max must be 19c, not 20c.
    assert PR.max_buy_price(0.215) == 0.19


def test_threshold_semantics_match_book_lines():
    samples = np.array([4, 5, 5, 6])
    # "over 4.5" == ">= 5" == "5+"
    assert PR.prob_at_least(samples, 4.5) == PR.prob_at_least(samples, 5) == 0.75
    # complementary: P(>=5) + P(<=4) = 1
    assert PR.prob_at_least(samples, 5) + np.mean(samples <= 4) == 1.0


def test_implied_points_sign_convention():
    # spread_line > 0 means home favored (nflverse); total 45, home -3 -> 24 / 21
    assert implied_points(3.0, 45.0, True) == 24.0
    assert implied_points(3.0, 45.0, False) == 21.0


def test_eb_strength_recovers_known_concentration():
    rng = np.random.default_rng(0)
    m, k_true = 0.2, 30.0
    rates = rng.beta(m * k_true, (1 - m) * k_true, 4000)
    n = rng.integers(20, 200, 4000)
    succ = rng.binomial(n, rates)
    k_hat = _eb_strength(succ, n, m, 1, 5000)
    assert 22 < k_hat < 40


def test_bootstrap_sums_mean():
    rng = np.random.default_rng(1)
    league = np.array([0.0, 10.0])
    counts = np.full(20000, 3)
    s = _bootstrap_sums(rng, counts, None, None, league, 1.0)
    assert math.isclose(s.mean(), 15.0, rel_tol=0.02)
    assert _bootstrap_sums(rng, np.zeros(5, int), None, None, league, 1.0).sum() == 0

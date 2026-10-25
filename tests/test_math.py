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


def test_unavailable_ids_uses_latest_week_and_respects_trades():
    import pandas as pd
    from gambit.data import unavailable_ids
    rs = pd.DataFrame([
        # IR since week 4 and still IR in the latest published week -> out
        dict(week=4, gsis_id="thornton", status="ACT"), dict(week=6, gsis_id="thornton", status="RES"),
        # came off IR before the slate week -> available
        dict(week=5, gsis_id="kirk", status="RES"), dict(week=7, gsis_id="kirk", status="ACT"),
        # traded: two rows in the same week, ACT on the new team wins
        dict(week=7, gsis_id="traded", status="TRD"), dict(week=7, gsis_id="traded", status="ACT"),
        # practice squad is not an injury status
        dict(week=7, gsis_id="ps", status="DEV"),
        # a later week than the slate is ignored
        dict(week=6, gsis_id="future", status="ACT"), dict(week=9, gsis_id="future", status="RES"),
    ])
    out = unavailable_ids(rs, 7)
    assert set(out) == {"thornton"}
    assert out["thornton"] == "RES"


def test_league_draws_recency_weighting():
    from gambit.model import _league_draws
    rng = np.random.default_rng(3)
    league = np.array([0.0, 10.0])
    cdf = np.cumsum([1.0, 3.0]) / 4.0          # weight 3x on the second value
    d = _league_draws(rng, league, 40000, cdf)
    assert math.isclose(d.mean(), 7.5, rel_tol=0.02)
    assert math.isclose(_league_draws(rng, league, 40000).mean(), 5.0, rel_tol=0.02)


def test_efficiency_multiplier_direction():
    """A team implied below its norm must get fewer yards per catch, never more."""
    from gambit import model as M
    st = M.State(cutoff=(2026, 7), params=M.Params())
    st.eff_implied = {"catch": 0.01, "rush": 0.0}
    st.team_implied_norm = {"KC": 24.6}
    d = float(np.clip(19.75 - st.team_implied_norm["KC"], -st.params.eff_clip, st.params.eff_clip))
    assert np.exp(st.eff_implied["catch"] * d) < 1.0
    assert np.exp(st.eff_implied["rush"] * d) == 1.0


def test_questionable_volume_moves_to_group_only_when_he_sits():
    import pandas as pd
    from gambit.model import _group_scaled_shares
    P = pd.DataFrame({"a_tgt": [30.0, 20.0, 10.0, 15.0], "b_tgt": [70.0, 80.0, 90.0, 85.0],
                      "group": ["WR", "WR", "WR", "TE"]}, index=list("abcd"))
    n = 4000
    present = np.ones((4, n), bool)
    present[0, : n // 2] = False                     # WR "a" sits in the first half of sims
    rng = np.random.default_rng(0)
    out, other = _group_scaled_shares(rng, P, "tgt", {"WR": 0.6, "TE": 0.2}, n, None, present)
    assert np.all(out[0, : n // 2] == 0)            # zero volume when he sits
    np.testing.assert_allclose(out[:3].sum(axis=0), 0.6, atol=1e-9)   # WR group keeps its share every sim
    np.testing.assert_allclose(out[3], 0.2, atol=1e-9)                # other groups untouched
    assert out[1, : n // 2].mean() > out[1, n // 2:].mean() * 1.3     # teammates absorb it


def test_own_samples_conditions_on_playing():
    from gambit.model import own_samples
    d = {"rec_yds": np.array([0.0, 50.0, 0.0, 70.0]), "played": np.array([False, True, False, True])}
    np.testing.assert_array_equal(own_samples(d, "rec_yds"), [50.0, 70.0])
    assert own_samples({"x": np.arange(3)}, "x").tolist() == [0, 1, 2]
    assert own_samples(d, "missing") is None


def test_availability_fit_orders_listing_practice_and_prev_game():
    import pandas as pd
    from gambit import availability as A
    rng = np.random.default_rng(1)
    full, lim, dnp = ("Full Participation in Practice", "Limited Participation in Practice",
                      "Did Not Participate In Practice")
    rows = []
    for i in range(6000):
        q = rng.random() < 0.3
        prac = rng.choice([full, lim, dnp]) if q else None
        prev = rng.random() < 0.8
        role = rng.uniform(0.05, 1.0)
        z = -0.5 + 5 * role + 1.5 * prev - (1.3 if q else 0) - {full: 0, lim: 0.4, dnp: 1.5, None: 0}[prac]
        rows.append(dict(report_status="Questionable" if q else None, practice_status=prac, role=role,
                         played_prev=prev, y=rng.random() < 1 / (1 + np.exp(-z)), t=40510))
    av = A.fit(pd.DataFrame(rows))
    probe = pd.DataFrame(dict(report_status=["Questionable"] * 3 + [None, None],
                              practice_status=[full, lim, dnp, None, None], role=[0.7] * 5,
                              played_prev=[True, True, True, True, False]))
    p = av.prob(probe)
    assert p[0] > p[1] > p[2]          # Full > Limited > DNP
    assert p[3] > p[0]                 # unlisted > Questionable
    assert p[3] > p[4]                 # played last game > missed it
    # walk-forward: rows at or after the cutoff week are not used
    assert A.fit_before(pd.DataFrame(rows), (2025, 10)).n == 0
    assert A.fit_before(pd.DataFrame(rows), (2025, 11)).n == 6000


def test_window_features_counts_missed_games_as_zero():
    import pandas as pd
    from gambit import availability as A
    snaps = pd.DataFrame(dict(player_id=["a", "a", "a", "b"], posteam="KC", season=2025,
                              week=[3, 4, 5, 3], offense_pct=[0.9, 0.9, 0.9, 0.6]))
    f = A.window_features(snaps, "KC", 2025 * 20 + 6)
    assert f.loc["a", "role"] == pytest.approx(0.9) and f.loc["a", "played_prev"]
    assert f.loc["b", "role"] == pytest.approx(0.2) and not f.loc["b", "played_prev"]


def test_pregame_status_ignores_target_week_game_day_inactives():
    import pandas as pd
    from gambit.data import unavailable_ids
    rs = pd.DataFrame(dict(gsis_id=["x", "x", "y", "y", "z"], week=[5, 6, 5, 6, 6],
                           status=["INA", "INA", "RES", "INA", "RES"], team="KC"))
    assert set(unavailable_ids(rs, 6)) == {"x", "y", "z"}            # after the inactive list is out
    # before kickoff: Week 6 INA is unknown; an older status must not stand in for it
    assert set(unavailable_ids(rs, 6, pregame=True)) == {"z"}

# EV+ Gambit v3

NFL player-prop pricer for DraftKings Predictions contracts. Five markets: rushing yards,
anytime TD, passing yards, receptions, receiving yards.

It answers one question per contract: **what is the most I should pay for YES (or NO) after
DK's per-contract fee?** You type the DK price; it does the rest.

No paid data. Play-by-play, closing spreads/totals, snap counts and injury reports come from
[nflverse](https://github.com/nflverse) via `nflreadpy`; kickoff wind from Open-Meteo.

## The model, in one paragraph

Market spread and total → each team's implied points → fitted play volume, pass share
(with an estimated wind effect) and expected touchdowns. Each player's carry and target
share is an empirical-Bayes posterior: recency-weighted, shrunk toward a *role* prior
(RB1, WR2…) with shrinkage strength estimated by method of moments, plus estimated
game-to-game volatility. A player ruled OUT has his volume redistributed within his
position group. Yards are bootstrapped per play from the player's own history mixed with
the league pool (mixing weight estimated; RB yards per carry is mostly noise, so RBs lean
on the league pool). TDs = simulated volume × the player's expected-TD rate per
opportunity (goal-line role lives here), scaled so the team total matches the market.
This-game shares are drawn Dirichlet around long-run shares so estimated game-to-game
volatility survives the position-group constraint. A per-market logistic calibration
(fit on all graded 2025 thresholds; slopes 0.91–1.01, i.e. small corrections) is validated
on 2026.

## Does it work?

`results/backtest_summary.csv` holds the walk-forward backtest: fit only on data before each
week, simulate with that game's closing lines, grade against official box scores. Compared
on identical rows to (a) the "how many times has he hit it" rate and (b) the August v2 math.
On the **2026 holdout** (weeks 1–6, contested thresholds), 95% cluster-bootstrap CIs:

| Market | vs hit-rate (Brier diff) | vs v2 math | Verdict |
|---|---|---|---|
| Anytime TD | −0.022 [−0.029, −0.015] | −0.027 [−0.035, −0.019] | better than both |
| Rushing yards | −0.036 [−0.062, −0.010] | −0.014 [−0.036, +0.009] | better than hit-rate |
| Receptions | −0.012 [−0.024, −0.001] | −0.009 [−0.019, +0.002] | better than hit-rate (narrowly) |
| Receiving yards | −0.004 [−0.020, +0.012] | −0.008 [−0.024, +0.009] | not significant |
| Passing yards | −0.005 [−0.025, +0.015] | −0.020 [−0.038, −0.001] | better than v2 math only |

Holdout reliability (`results/reliability_2026_holdout.csv`) is within a few points of
observed in every well-populated bin; the thin top bin for receiving yards (n=83,
predicted 0.86 vs observed 0.69) is the remaining weak spot.

This is probability *quality*, not profit. Beating a hit-rate is not beating the market.
The only evidence of edge against DK is closing-line value, which the Ledger tab tracks.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
python -m pytest -q tests
python -m gambit.backtest 2026 1 6          # walk-forward backtest for a season/week range
python -m gambit.calibrate backtest_2025_2_18.parquet   # refit calibration (do this at season boundaries)
```

## Known limits (honest list)

- QB change is priced only through the market's team total, not through receiver efficiency.
- Near the median, receiving-yard and rush-yard distributions remain somewhat too peaked
  (a contested-only calibration would want slopes ~0.64–0.69); the shipped calibration does
  not correct this because doing so damaged the tails.
- Role ranks are assigned from the same data used for the posterior (mild winner's curse).
- Positions are taken from the latest available week (negligible leakage in backtests).
- Weather: only wind enters; temperature/precipitation are displayed, not modeled.
- Backtest uses actual game wind and snap-based actives (mild optimism vs. live use).
- Graded rows are selected by pre-game box-score volume (last 3 games), never by model output.
- Open-Meteo rate-limits shared IPs (Streamlit Cloud included): when the forecast fails, type
  the wind in. Retractable-roof venues default to closed (wind 0) until the roof is announced.
- Streamlit Cloud storage is ephemeral: download the ledger CSV after each session.
- NCAAF and NBA are out of scope for v3.

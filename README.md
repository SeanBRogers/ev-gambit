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
(fit on all graded 2025 thresholds; slopes 0.91–1.03, i.e. small corrections) is validated
on 2026. Players on IR/PUP/NFI, suspended, exempt or inactive are removed by default from the
weekly roster status, and the market total also scales yards per catch (v3.1, below). Before
inactives, every other player plays in each simulation with an estimated probability (v3.2, below).

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

## v3.1 (Oct 25, 2026) — what changed and what it bought

Three diagnoses from live slates, each verified in the code and data before fixing:

1. **Injured-reserve players were never removed.** Default outs came only from the weekly
   injury report, which never lists IR/PUP/NFI, suspended or exempt players, and the roster
   is "on the field in the last 3 team games." So a player hurt in that window kept his old
   share (KC's Tyquan Thornton, IR since Week 4: 3.4 targets in the Week 7 export; WAS's Jaylin
   Lane in Week 6). Fix: weekly roster status from nflverse (`load_roster_status`); anything
   not ACT is out by default, and the Slate tab says why. The backtest hid this because it
   removed players using the target game's *actual* snaps; it now has `info="report"` /
   `"report+status"` modes that use only what is public before kickoff.
2. **Market totals moved play volume but not yards per play.** Implied points fed plays
   (+0.3 per point) and TDs only, so a QB implied for 27 and one implied for 17 got the same
   per-catch efficiency. Data: within team-season, yards per catch rise ~0.9%/implied point
   (t≈4) and yards per carry ~1.4%/pt. Fix: catches scale with implied points relative to the
   team's own norm. The rushing analogue is estimated but **not used** — it did not improve
   RB props out of sample.
3. **League yardage pool ignored recency** (2026 rush yds/carry 4.04 vs 4.28–4.33 in 2024–25).
   Fix: sampled with the model's recency weights. Neutral in 2025, slight help in 2026.

Diagnosed and **not** changed: target concentration. Posterior shares for WR1–WR3 match raw
shares within 0.1 pt league-wide; depth-WR inflation is ~0.8 pt of team targets, and extending
role priors to rank 6 did not help.

`results/v31_vs_v3.csv`: as shipped (live information, calibrated; calibration refit on 2025),
2026 holdout weeks 1–6, Brier v3.1 − v3 with 95% cluster-bootstrap CI, contested thresholds.

**Correction (v3.2):** the first version of this table was leaky. Its "live" mode read the target
week's roster status, which includes the game-day inactive list (`INA`): nflverse shows no INA
rows for games not yet played (Week 7 2026: 0 for the 10 unplayed teams, 6–7 per finished team),
so that list is not available when the injury report is. Re-run without it:

| Market | Brier diff (leak-free) | First published | Bias on contested lines, v3 → v3.1 |
|---|---|---|---|
| Rushing yards | −0.0044 [−0.012, +0.003] | −0.0105 [−0.021, −0.001] | −1.8 → −0.1 pts |
| Receptions | −0.0043 [−0.008, −0.001] | −0.0057 | −9.5 → −6.8 pts |
| Receiving yards | −0.0054 [−0.010, −0.001] | −0.0050 | −8.4 → −5.7 pts |
| Passing yards | −0.0044 [−0.008, −0.001] | −0.0043 | −6.0 → −4.9 pts |
| Anytime TD | −0.0006 [−0.002, +0.001] | −0.0009 | −1.9 → −1.0 pts |

The rushing improvement is no longer significant. The other markets hold. 2025 points the same
way in pass yards and receptions (significant) and receiving yards (not significant). It is flat
in rushing yards and TDs. Still probability quality, not profit.

## v3.2 (Oct 25, 2026) — who actually plays

Before kickoff the app knows the injury report and weekly roster status, but not the game-day
inactive list. v3.1 assumed everyone not ruled out plays. Among non-QB players on the simulated
roster (2025 and 2026 Weeks 1–6, pre-kickoff information only), the share who took an offensive
snap was:

| Group | 2025 | 2026 |
|---|---|---|
| Listed Questionable | 66% | 63% |
| Not listed, missed the team's previous game | 37% | 36% |
| Not listed, played the previous game | 94% | 92% |

The second group (healthy scratches, players coming back, depth players rotated out) left more
target share on the table than the Questionable group did. Fix (`gambit/availability.py`): two
logistic regressions, fit only on team-weeks before the slate.
- Questionable players: practice status plus recent snap share.
- Everyone else not ruled out: practice status, played the previous game, recent snap share, and
  their interaction.

Each player plays or sits in each simulation. When he sits, his volume goes to his position group.
His own props are priced on the simulations in which he plays.

Out of sample (fit before each season, scored on it), P(plays) Brier is 0.075 vs 0.128 for
"everyone plays" in 2025, and 0.063 vs 0.096 in 2026. One pooled model was tried first. On
Questionable players it was worse than a constant rate (0.238 vs 0.224). It also over-stated how
often a Questionable starter plays, because its snap-share slope was set by depth players.
Props scored the same under either model (all differences within ±0.001).

`results/v32_vs_v31.csv`: Brier v3.2 − v3.1, same calibration, live information, contested
thresholds, 95% cluster-bootstrap CI.

| Market | 2026 holdout | 2025 | Bias 2026, v3.1 → v3.2 |
|---|---|---|---|
| Receptions | −0.0045 [−0.007, −0.002] | −0.0024 [−0.004, −0.001] | −6.8 → −3.4 pts |
| Rushing yards | −0.0021 [−0.009, +0.005] | −0.0046 [−0.009, −0.001] | −0.1 → +4.5 pts |
| Receiving yards | −0.0010 [−0.004, +0.002] | −0.0008 [−0.003, +0.001] | −5.7 → −2.9 pts |
| Passing yards | −0.0002 [−0.001, +0.001] | +0.0001 [−0.000, +0.001] | −4.7 → −4.6 pts |
| Anytime TD | +0.0003 [−0.001, +0.001] | −0.0003 [−0.001, +0.000] | −0.9 → +0.3 pts |

Receptions improve in both seasons. Rushing yards improve in 2025 only. The rest is unchanged.
Calibration was **not** refit: refitting it on v3.2's 2025 output was marginally worse on the
2026 holdout in four of five markets. Against v3 (`results/v32_vs_v3.csv`), 2026 receptions are
−0.0088 [−0.014, −0.004] and passing yards −0.0045 [−0.008, −0.001].

What this does **not** know: contract settlement when a player sits. The Price tab asks. A
sportsbook voids the prop. Whether DraftKings Predictions resolves it NO is not verified here.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
python -m pytest -q tests
python -m gambit.backtest 2026 1 6          # walk-forward backtest for a season/week range (snap-based actives)
python -m gambit.backtest 2026 1 6 report+status         # pre-kickoff information, everyone not ruled out plays (v3.1)
python -m gambit.backtest 2026 1 6 report+status+avail   # pre-kickoff information + availability model (v3.2, as shipped)
python -m gambit.calibrate backtest_2025_2_18.parquet   # refit calibration (do this at season boundaries)
```

## Known limits (honest list)

- **Live receiving markets still run a little low**: 2026 contested receptions −3.4 pts and
  receiving yards −2.9 pts after v3.2. Practice-squad elevations and players with no recent snaps
  are still invisible. A sitting player's volume all goes to modeled teammates in his group,
  which over-feeds RBs: 2026 contested rushing is now +4.5 pts.
- 2026 rushing is over-predicted when actives are known (+6 pts on contested lines); recency
  weighting only partly corrects it.
- Availability uses each player's last 3 *team* games. In Weeks 1–3 that window reaches into
  last season.
- Before this week's nflverse roster rows are published, last week's game-day inactives carry
  forward as OUT (89% of them did not play the next week either).
- QB quality is still priced only through the market total and his receivers' history — no
  QB-specific completion or depth effect.
- Near the median, receiving-yard and rush-yard distributions remain somewhat too peaked
  (a contested-only calibration would want slopes ~0.64–0.69); the shipped calibration does
  not correct this because doing so damaged the tails.
- Role ranks are assigned from the same data used for the posterior (mild winner's curse).
- Positions are taken from the latest available week (negligible leakage in backtests).
- Weather: only wind enters; temperature/precipitation are displayed, not modeled.
- The default backtest uses actual game wind and snap-based actives. Snap-based actives were
  **not** mild optimism: they hid the IR bug. Use `info="report+status+avail"` to grade live
  behavior. Pre-kickoff modes never read the target week's game-day inactive list.
- Graded rows are selected by pre-game box-score volume (last 3 games), never by model output.
- Open-Meteo rate-limits shared IPs (Streamlit Cloud included): when the forecast fails, type
  the wind in. Retractable-roof venues default to closed (wind 0) until the roof is announced.
- Streamlit Cloud storage is ephemeral: download the ledger CSV after each session.
- NCAAF and NBA are out of scope for v3.

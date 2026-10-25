"""
Walk-forward backtest.

For each target week: fit on data strictly before it, simulate every game with the
market's closing spread/total, then grade P(stat >= threshold) against official box
scores. No historical prop prices exist in free data, so this measures CALIBRATION and
SHARPNESS of probabilities (Brier score, log loss, reliability), not profit.

Compared against two baselines on the identical (player, game, threshold) rows:
  hit_rate : share of the player's last 8 games with stat >= threshold, (h+1)/(n+2)
             -- the "how many times has he hit it" approach
  old      : the August Gambit math (season mean, fixed sigma, its continuity
             correction, shrink-output-toward-50%)

Information set: roster = players on the field in the team's last 3 games, minus anyone
without an offensive snap in the target game (inactives are public ~90 min pre-kickoff;
snaps > 0 slightly overstates that knowledge for dressed-but-unused players). Actual game
wind is used where a forecast would be in live use. Both are mild optimism.

Usage:  python -m gambit.backtest 2025 2 18      (season, first week, last week)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data as D
from . import model as M

THRESHOLDS = {
    "rush_yds": [30, 40, 50, 60, 70, 80, 90, 100],
    "receptions": [2, 3, 4, 5, 6, 7, 8],
    "rec_yds": [30, 40, 50, 60, 70, 80, 90],
    "pass_yds": [175, 200, 225, 250, 275, 300],
    "anytime_td": [1],
}
STAT_COL = {"rush_yds": "rushing_yards", "receptions": "receptions", "rec_yds": "receiving_yards",
            "pass_yds": "passing_yards"}
OLD_SIGMA = {"rush_yds": 25.0, "receptions": 1.5, "rec_yds": 25.0, "pass_yds": 45.0}
# Relevance (would this prop plausibly be posted?) from PRE-GAME box scores only — mean over the
# player's last 3 games — never from model output, which would select rows the model likes.
RELEVANT = {"rush_yds": ("carries", 6), "receptions": ("targets", 2.5), "rec_yds": ("targets", 3),
            "pass_yds": ("attempts", 15), "anytime_td": ("touches", 2)}


def _old_prob(mean_pg, gp, thr, market, td_rate=None):
    """Replicates the August Gambit evaluate_prop / anytime-TD logic for 'thr+' (line thr-0.5)."""
    def shrink(p, n, m=0.5, w=3):
        return (p * n + w * m) / (n + w)
    if market == "anytime_td":
        return shrink(td_rate, gp, 0.30)
    line = thr - 0.5
    p = 1 - norm.cdf(line + 0.5, loc=mean_pg, scale=OLD_SIGMA[market])
    return shrink(p, gp)


def run(season: int, w0: int, w1: int, n_sims: int = 4000, params: M.Params | None = None,
        info: str = "oracle"):
    """info = which players the simulation removes before kickoff:
      oracle         -> anyone without an offensive snap in the target game (v3 default; optimistic)
      report         -> that week's Out/Doubtful injury report only (what v3's app actually did live)
      report+status  -> report plus weekly roster status (IR/PUP/NFI/suspended/exempt/inactive)
    Grading is identical in every mode: only players who actually played are graded.
    """
    seasons = [season - 2, season - 1, season] if season >= 2026 else [season - 1, season]
    pbp = D.load_pbp(seasons, refresh_current=False)
    pos = D.load_positions(seasons)
    sch = D.load_schedule([season])
    stats = D.load_player_stats(seasons)
    stats["tds"] = stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)
    stats["t"] = stats["season"] * 20 + stats["week"]
    inj = D.load_injuries(season)
    snaps = D.load_snaps(seasons)
    rstat = D.load_roster_status(season) if info == "report+status" else None
    rows = []
    for week in range(w0, w1 + 1):
        t0 = time.time()
        st = M.fit_state(pbp, pos, cutoff=(season, week), params=params, snaps=snaps)
        games = sch[(sch["week"] == week) & sch["result"].notna() & sch["spread_line"].notna()]
        wk_stats = stats[(stats["season"] == season) & (stats["week"] == week)].set_index("player_id")
        hist = stats[stats["t"] < season * 20 + week]
        # Actives are public ~90 min before kickoff; proxy = had an offensive snap.
        played = set(snaps[(snaps["season"] == season) & (snaps["week"] == week)]["player_id"])
        wk_out = set(inj.loc[(inj["week"] == week) & inj["report_status"].isin(["Out", "Doubtful"]), "gsis_id"])
        if rstat is not None:
            wk_out |= set(D.unavailable_ids(rstat, week))
        for _, g in games.iterrows():
            outdoor = g["roof"] in ("outdoors", "open")
            wind = float(g["wind"]) if outdoor and pd.notna(g["wind"]) else 0.0
            for team, home in [(g["home_team"], True), (g["away_team"], False)]:
                imp = D.implied_points(g["spread_line"], g["total_line"], home)
                ts = g["spread_line"] if home else -g["spread_line"]
                if info == "oracle":
                    team_outs = tuple(p for p in st.roster.get(team, []) if p not in played)
                else:
                    team_outs = [p for p in st.roster.get(team, []) if p in wk_out]
                    # mirror the app: only the default starter (most weighted dropbacks) plays QB
                    qbs = [p for p in st.roster.get(team, []) if p not in team_outs
                           and st.players.loc[p, "group"] == "QB"]
                    if qbs:
                        starter = max(qbs, key=lambda p: st.players.loc[p, "n_db"])
                        team_outs += [p for p in qbs if p != starter]
                    team_outs = tuple(team_outs)
                sim = M.simulate_team(st, M.GameInputs(team, "", imp, ts, wind, team_outs),
                                      n=n_sims, seed=week * 1000 + sum(map(ord, team)))
                if "error" in sim.get("meta", {}):
                    continue
                for pid, d in sim.items():
                    if pid == "meta" or pid not in played:
                        continue
                    if pid in wk_stats.index:
                        actual = wk_stats.loc[pid]
                        if isinstance(actual, pd.DataFrame):
                            actual = actual.iloc[0]
                    else:  # on the field but no box-score row: every stat is 0
                        actual = pd.Series(0.0, index=list(STAT_COL.values()) + ["tds"])
                    h = hist[hist["player_id"] == pid].sort_values("t")
                    last8 = h.tail(8)
                    cur = h[h["season"] == season]
                    base_src = cur if len(cur) else h[h["season"] == season - 1]
                    d = dict(d)
                    d["anytime_td"] = d["tds"]
                    last3 = h.tail(3)
                    pre = {c: float(last3[c].fillna(0).mean()) if len(last3) else 0.0
                           for c in ["carries", "targets", "attempts"]}
                    pre["touches"] = pre["carries"] + pre["targets"]
                    for market, thrs in THRESHOLDS.items():
                        rel_key, rel_min = RELEVANT[market]
                        if pre[rel_key] < rel_min or market not in d:
                            continue
                        if market == "anytime_td":
                            outcome_val = actual["tds"]
                            hist_vals = last8["tds"].to_numpy()
                        else:
                            outcome_val = actual[STAT_COL[market]]
                            hist_vals = last8[STAT_COL[market]].to_numpy()
                        if pd.isna(outcome_val) or len(hist_vals) == 0:
                            continue
                        med8 = float(np.median(hist_vals))
                        for thr in thrs:
                            p = float(np.mean(d[market] >= thr))
                            hr = (np.sum(hist_vals >= thr) + 1) / (len(hist_vals) + 2)
                            if market == "anytime_td":
                                old = _old_prob(None, len(base_src), thr, market,
                                                td_rate=float(base_src["tds"].mean()) if len(base_src) else 0.0)
                            else:
                                mpg = float(base_src[STAT_COL[market]].mean()) if len(base_src) else 0.0
                                old = _old_prob(mpg, len(base_src), thr, market)
                            contested = (market == "anytime_td") or (0.75 * med8 <= thr <= 1.25 * med8 + 0.5)
                            rows.append(dict(season=season, week=week, game_id=g["game_id"], team=team,
                                             player_id=pid, market=market, thr=thr, p_model=p,
                                             p_hit_rate=hr, p_old=old, y=int(outcome_val >= thr),
                                             contested=contested))
        print(f"{season} wk{week:02d}: {len(games)} games, rows so far {len(rows)}, {time.time()-t0:.1f}s",
              flush=True)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    eps = 1e-4
    for (market, contested), d in df.groupby(["market", "contested"]):
        r = dict(market=market, subset="contested" if contested else "all_other", n=len(d),
                 base_rate=d["y"].mean())
        for m in ["p_model", "p_hit_rate", "p_old"]:
            p = d[m].clip(eps, 1 - eps)
            r[f"brier_{m[2:]}"] = float(np.mean((p - d["y"]) ** 2))
            r[f"logloss_{m[2:]}"] = float(-np.mean(d["y"] * np.log(p) + (1 - d["y"]) * np.log(1 - p)))
            r[f"meanp_{m[2:]}"] = float(d[m].mean())
        out.append(r)
    return pd.DataFrame(out)


def paired_diff(df: pd.DataFrame, a: str = "p_model", b: str = "p_hit_rate",
                B: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Brier(a) - Brier(b) with a cluster bootstrap over player-games.

    Rows sharing a player-game (several thresholds) are not independent, so resampling
    rows would understate the SE. Negative = `a` is better.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (market, contested), d in df.groupby(["market", "contested"]):
        d = d.assign(cl=d["game_id"] + "|" + d["player_id"],
                     da=(d[a] - d["y"]) ** 2 - (d[b] - d["y"]) ** 2)
        per = d.groupby("cl")["da"].agg(["sum", "size"])
        s, n = per["sum"].to_numpy(), per["size"].to_numpy()
        idx = rng.integers(0, len(s), (B, len(s)))
        boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
        est = s.sum() / n.sum()
        out.append(dict(market=market, subset="contested" if contested else "all_other",
                        clusters=len(s), diff=est, se=boots.std(ddof=1),
                        ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5)))
    return pd.DataFrame(out)


def reliability(df: pd.DataFrame, col: str = "p_model", bins: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.cut(d[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    return d.groupby(["market", "bin"], observed=True).agg(n=("y", "size"), predicted=(col, "mean"),
                                                            observed=("y", "mean")).reset_index()


if __name__ == "__main__":
    season, w0, w1 = (int(x) for x in sys.argv[1:4])
    info = sys.argv[4] if len(sys.argv) > 4 else "oracle"
    if info not in ("oracle", "report", "report+status"):
        raise SystemExit("info must be oracle | report | report+status")
    res = run(season, w0, w1, info=info)
    suffix = "" if info == "oracle" else "_" + info.replace("+", "_")
    res.to_parquet(f"backtest_{season}_{w0}_{w1}{suffix}.parquet", index=False)
    pd.set_option("display.width", 250)
    print(score(res).round(4).to_string(index=False))

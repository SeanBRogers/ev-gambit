"""
Generative prop model.

    market lines ──► team volume (plays, pass share, TD intensity)
                       │
    player shares ─────┼──► carries / targets / TD allocation   (multinomial)
    (empirical Bayes)  │
    per-play pools ────┴──► yards (bootstrap mixture: player ⊕ league)

Every parameter is estimated from data before the cutoff week. Shrinkage
strength (k) is estimated by method of moments, not asserted. Parameter
uncertainty is propagated: each simulation draws the player's share and
catch rate from its posterior Beta, so thin evidence widens the predictive
distribution instead of being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D

POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}


def _group(pos) -> str:
    return POS_GROUP.get(pos, "OTH") if isinstance(pos, str) else "OTH"


def time_index(season: int, week: int) -> int:
    return int(season) * 20 + int(week)


@dataclass
class Params:
    decay: float = 0.88          # weight multiplier per week back
    season_decay: float = 0.6    # extra multiplier per season boundary crossed
    lookback_active: int = 3     # team games used to define the active roster
    min_k: float = 2.0
    max_k: float = 2000.0
    # Market-implied points move yards per play, not just play count. Estimated within
    # team-season (the matchup part a player's own history cannot carry). v3 used the total
    # only for plays (+0.3/pt) and TDs, which compressed every QB toward ~200-250 yards.
    implied_eff: bool = True
    # Pools that get the adjustment. Rushing is estimated (+1.4%/pt, t~5 at team level) but did
    # not improve RB rushing-yard props out of sample (2025: -0.0002 Brier, CI spans 0; 2026
    # holdout: +0.0018, CI spans 0), so only the passing game uses it.
    implied_eff_pools: tuple = ("catch",)
    # League yardage pool sampled with the same recency weights as everything else, so the
    # current season's environment (2026 rush yds/carry 4.04 vs 4.28-4.33 in 2024-25) counts.
    league_recency: bool = True
    max_role_rank: int = 4       # role priors for ranks 1..max_role_rank; deeper players share the last
    eff_clip: float = 10.0       # cap on |implied - team norm| in points


# ---------------------------------------------------------------------------
# Small estimation helpers
# ---------------------------------------------------------------------------

def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    w = np.ones(len(y)) if w is None else w
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(np.sum(w * resid ** 2) / np.sum(w) * len(y) / dof)
    xtwx_inv = np.linalg.pinv((X * w[:, None]).T @ X)
    se = np.sqrt(np.diag(xtwx_inv) * sigma2 * np.sum(w) / len(y))
    return beta, se, resid, sigma2


def _eb_strength(successes, trials, prior_mean, min_k, max_k) -> float:
    """Beta-binomial concentration k by method of moments.

    prior_mean may be a scalar or one value per unit (role-specific priors).
    var_obs(rate - m) = var_true + E[m(1-m)/n]  →  k = E[m(1-m)] / var_true - 1.
    """
    trials = np.asarray(trials, float)
    ok = trials > 0
    if ok.sum() < 5:
        return max_k
    r = np.asarray(successes, float)[ok] / trials[ok]
    n = trials[ok]
    m = np.broadcast_to(np.asarray(prior_mean, float), trials.shape)[ok]
    var_obs = np.average((r - m) ** 2, weights=n)
    noise = np.average(m * (1 - m) / n, weights=n)
    var_true = var_obs - noise
    if var_true <= 1e-9:
        return max_k
    return float(np.clip(np.average(m * (1 - m), weights=n) / var_true - 1, min_k, max_k))


def _eb_mean_strength(values_by_unit: list[np.ndarray], min_k, max_k) -> float:
    """For per-play outcomes (yards): k = within-unit var / between-unit true var."""
    vs = [v for v in values_by_unit if len(v) >= 2]
    if len(vs) < 5:
        return max_k
    means = np.array([v.mean() for v in vs])
    ns = np.array([len(v) for v in vs])
    within = np.average([v.var(ddof=1) for v in vs], weights=ns - 1)
    grand = np.average(means, weights=ns)
    var_obs = np.average((means - grand) ** 2, weights=ns)
    var_true = var_obs - np.average(within / ns, weights=ns)
    if var_true <= 1e-9:
        return max_k
    return float(np.clip(within / var_true, min_k, max_k))


# ---------------------------------------------------------------------------
# Fitted state at a cutoff
# ---------------------------------------------------------------------------

@dataclass
class State:
    cutoff: tuple[int, int]
    params: Params
    plays_beta: np.ndarray = None
    plays_sigma: float = None
    pass_beta: np.ndarray = None
    pass_tau: float = None
    td_beta: np.ndarray = None
    wind_ypa_slope: float = 0.0
    wind_ypa_se: float = 0.0
    league_ypa: float = 7.0
    team_plays_eff: dict = field(default_factory=dict)
    team_pass_eff: dict = field(default_factory=dict)
    team_group_share: dict = field(default_factory=dict)   # (team, kind) -> {group: share}
    players: pd.DataFrame = None                           # per-player posteriors
    pools: dict = field(default_factory=dict)              # (player_id, pool) -> (values, weights)
    league_pools: dict = field(default_factory=dict)       # (group, pool) -> values
    k_yards: dict = field(default_factory=dict)            # (group, pool) -> k
    roster: dict = field(default_factory=dict)             # team -> list of player_ids (active)
    psi: dict = field(default_factory=dict)                # (kind, group) -> game-to-game share concentration
    eff_sigma: dict = field(default_factory=dict)          # pool -> sd of log team game-efficiency factor
    eff_implied: dict = field(default_factory=dict)        # pool -> d log(yards/play) per implied point
    team_implied_norm: dict = field(default_factory=dict)  # team -> recency-weighted mean implied points
    league_implied: float = 22.0
    league_pool_cdf: dict = field(default_factory=dict)    # (group, pool) -> cumulative recency weights
    diagnostics: dict = field(default_factory=dict)


def fit_state(pbp: pd.DataFrame, positions: pd.DataFrame, cutoff: tuple[int, int],
              params: Params | None = None, snaps: pd.DataFrame | None = None) -> State:
    """Fit everything using only games strictly before `cutoff` = (season, week)."""
    params = params or Params()
    t_cut = time_index(*cutoff)
    pbp = pbp[(pbp["season"] * 20 + pbp["week"]) < t_cut]
    tg = D.build_team_games(pbp)
    op = D.build_opportunities(pbp)
    xtd = D.league_xtd(op)

    st = State(cutoff=cutoff, params=params)
    tg["t"] = tg["season"] * 20 + tg["week"]
    tg["w"] = params.decay ** (t_cut - tg["t"]) * params.season_decay ** (cutoff[0] - tg["season"])
    op["t"] = op["season"] * 20 + op["week"]
    op["w"] = params.decay ** (t_cut - op["t"]) * params.season_decay ** (cutoff[0] - op["season"])

    # ---------------- team volume ----------------
    Xp = np.column_stack([np.ones(len(tg)), tg["implied"], tg["team_spread"]])
    b, se, resid, s2 = _wls(Xp, tg["plays"].to_numpy(float))
    st.plays_beta = b
    st.diagnostics["plays"] = dict(coef=b.round(3).tolist(), se=se.round(3).tolist(),
                                   names=["const", "implied", "team_spread"])
    tg["plays_resid"] = resid

    q = (tg["dropbacks"] + 0.5) / (tg["plays"] + 1.0)
    lq = np.log(q / (1 - q))
    Xq = np.column_stack([np.ones(len(tg)), tg["team_spread"], tg["wind_eff"]])
    bq, seq, rq, s2q = _wls(Xq, lq.to_numpy(), tg["plays"].to_numpy(float))
    st.pass_beta = bq
    st.diagnostics["pass_share_logit"] = dict(coef=bq.round(4).tolist(), se=seq.round(4).tolist(),
                                              names=["const", "team_spread", "wind_mph"])
    tg["pass_resid"] = rq

    # team effects: decayed EB shrinkage of residual means
    def team_effects(col, noise_var):
        eff = {}
        g = tg.groupby("posteam")
        means = g.apply(lambda d: np.average(d[col], weights=d["w"]), include_groups=False)
        var_between = max(np.var(means) - noise_var / max(g.size().mean(), 1), 1e-6)
        k = noise_var / var_between
        for team, d in g:
            eff[team] = float(np.sum(d["w"] * d[col]) / (np.sum(d["w"]) + k))
        return eff, k

    st.team_plays_eff, k_pl = team_effects("plays_resid", s2)
    pass_noise = float(np.average(1.0 / (tg["plays"] * q * (1 - q)), weights=tg["plays"]))
    st.team_pass_eff, k_pa = team_effects("pass_resid", s2q)
    # Residual variance minus the between-team part (now carried by team effects), and for
    # pass share also minus binomial sampling variance -> per-game script noise.
    st.plays_sigma = float(np.sqrt(max(s2 - s2 / k_pl, 1.0)))
    st.pass_tau = float(np.sqrt(max(s2q - pass_noise - s2q / k_pa, 0.01)))
    st.diagnostics["team_effect_k"] = dict(plays=round(k_pl, 1), pass_share=round(k_pa, 1))

    Xt = np.column_stack([np.ones(len(tg)), tg["implied"]])
    bt, set_, rt, s2t = _wls(Xt, tg["off_tds"].to_numpy(float))
    st.td_beta = bt
    disp = s2t / max(tg["off_tds"].mean(), 1e-6)
    st.diagnostics["off_tds"] = dict(coef=bt.round(4).tolist(), se=set_.round(4).tolist(),
                                     names=["const", "implied"], var_to_mean=round(float(disp), 3))

    # wind → yards per attempt (outdoor games)
    od = tg[(tg["attempts"] > 10)]
    ypa = od["pass_yards"] / od["attempts"]
    Xw = np.column_stack([np.ones(len(od)), od["wind_eff"]])
    bw, sew, *_ = _wls(Xw, ypa.to_numpy(float), od["attempts"].to_numpy(float))
    st.league_ypa = float(np.average(ypa, weights=od["attempts"]))
    st.wind_ypa_slope, st.wind_ypa_se = float(bw[1]), float(sew[1])
    st.diagnostics["wind_ypa"] = dict(slope_per_mph=round(bw[1], 4), se=round(sew[1], 4),
                                      league_ypa=round(st.league_ypa, 3))

    n_targets = int((op["kind"] == "target").sum())
    st.no_receiver_rate = float(1 - n_targets / max(tg["attempts"].sum(), 1))
    st.league_catch = float(op.loc[op["kind"] == "target", "caught"].mean())

    # ---------------- player layer ----------------
    pos = positions.set_index("player_id")
    op["group"] = op["player_id"].map(pos["position"]).map(_group).fillna("OTH")

    # team denominators per game
    den = tg.set_index(["game_id", "posteam"])[["designed_runs", "attempts"]]
    tm_xtd = op[op["kind"].isin(["carry", "scramble", "target"])].merge(xtd, on=["kind", "ybucket"], how="left")
    team_xtd = tm_xtd.groupby(["game_id", "posteam"])["p_td"].sum().rename("team_xtd")
    den = den.join(team_xtd, how="left").fillna({"team_xtd": 0.0})

    # per player-game opportunity counts
    pg = op.groupby(["player_id", "game_id", "posteam", "t", "w", "group"], as_index=False).agg(
        carries=("kind", lambda s: int((s == "carry").sum())),
        scrambles=("kind", lambda s: int((s == "scramble").sum())),
        targets=("kind", lambda s: int((s == "target").sum())),
        dropbacks=("kind", lambda s: int((s == "dropback").sum())),
    )
    catches = op[op["kind"] == "target"].groupby(["player_id", "game_id"])["caught"].sum()
    sacks = op[op["kind"] == "dropback"].groupby(["player_id", "game_id"])["sack"].sum()
    scr_db = op[op["kind"] == "dropback"].groupby(["player_id", "game_id"])["qb_scramble"].sum()
    pxtd = tm_xtd.groupby(["player_id", "game_id"])["p_td"].sum()
    pg = pg.set_index(["player_id", "game_id"])
    pg["catches"] = catches.reindex(pg.index).fillna(0)
    pg["sacks"] = sacks.reindex(pg.index).fillna(0)
    pg["db_scrambles"] = scr_db.reindex(pg.index).fillna(0)
    pg["xtd"] = pxtd.reindex(pg.index).fillna(0)
    pg = pg.reset_index()
    if snaps is not None and len(snaps):
        # games a player was on the field with zero opportunities still count in his denominator
        sn = snaps[(snaps["season"] * 20 + snaps["week"]) < t_cut]
        sn = sn[sn["game_id"].isin(set(tg["game_id"]))]
        have = set(zip(pg["player_id"], pg["game_id"]))
        add = sn[[(p, g) not in have for p, g in zip(sn["player_id"], sn["game_id"])]].copy()
        add["group"] = add["player_id"].map(pos["position"]).map(_group).fillna("OTH")
        add["t"] = add["season"] * 20 + add["week"]
        add["w"] = params.decay ** (t_cut - add["t"]) * params.season_decay ** (cutoff[0] - add["season"])
        for c in ["carries", "scrambles", "targets", "dropbacks", "catches", "sacks", "db_scrambles", "xtd"]:
            add[c] = 0
        pg = pd.concat([pg, add[pg.columns.intersection(add.columns)]], ignore_index=True)
    pg = pg.join(den, on=["game_id", "posteam"])

    # weighted sums per player
    def wsum(col):
        return (pg[col] * pg["w"]).groupby(pg["player_id"]).sum()

    P = pd.DataFrame({
        "c_car": wsum("carries"), "n_car": wsum("designed_runs"),
        "c_tgt": wsum("targets"), "n_tgt": wsum("attempts"),
        "c_cat": wsum("catches"),
        "c_xtd": wsum("xtd"), "n_xtd": wsum("team_xtd"),
        "c_sack": wsum("sacks"), "c_scr": wsum("db_scrambles"), "n_db": wsum("dropbacks"),
    })
    last = pg.sort_values("t").groupby("player_id").tail(1).set_index("player_id")
    P["team"] = last["posteam"]
    P["last_t"] = last["t"]
    P["group"] = last["group"]
    P["name"] = P.index.map(pos["display_name"]) if "display_name" in pos else P.index
    P["position"] = P.index.map(pos["position"])

    # ---------------- active roster (players on the field in the team's last N games) ----------------
    for team, d in tg.groupby("posteam"):
        recent = d.sort_values("t").tail(params.lookback_active)["game_id"]
        ids = pg[(pg["posteam"] == team) & pg["game_id"].isin(recent)]["player_id"].unique()
        st.roster[team] = [i for i in ids if P.loc[i, "team"] == team and P.loc[i, "group"] != "OTH"]
    on_roster = {p for ids in st.roster.values() for p in ids}

    # raw (unweighted) per-player counts for MoM prior strength
    raw = pg.groupby("player_id")[["carries", "designed_runs", "targets", "attempts",
                                   "catches", "xtd", "team_xtd", "sacks", "db_scrambles", "dropbacks"]].sum()

    st.diagnostics["k"] = {}
    # Shares: prior mean depends on ROLE (rank within team position group: RB1, RB2, WR1...),
    # not just position. A position-wide prior pulls fringe players up toward starter-level
    # shares and, after group normalization, steals volume from starters.
    P["on_roster"] = P.index.isin(on_roster)
    for share, c, n in [("car", "c_car", "n_car"), ("tgt", "c_tgt", "n_tgt")]:
        x = P[c] / P[n].replace(0, np.nan)
        max_r = int(params.max_role_rank)
        rk = (x.where(P["on_roster"]).groupby([P["team"], P["group"]])
              .rank(ascending=False, method="first").clip(upper=max_r))
        P[f"rank_{share}"] = rk
        P[f"m_{share}"] = np.nan
        P[f"k_{share}"] = np.nan
        for grp in ["QB", "RB", "WR", "TE", "OTH"]:
            gmask = P["group"] == grp
            if not gmask.any():
                continue
            m_grp = float(np.clip(P.loc[gmask, c].sum() / max(P.loc[gmask, n].sum(), 1e-9), 1e-4, 0.9))
            P.loc[gmask, f"m_{share}"] = m_grp
            role_means = {}
            prev = None
            for r_ in range(1, max_r + 1):
                rm = gmask & (P[f"rank_{share}"] == r_)
                if rm.sum() >= 5 and P.loc[rm, n].sum() > 0:
                    role_means[r_] = float(np.clip(P.loc[rm, c].sum() / P.loc[rm, n].sum(), 1e-4, 0.9))
                    prev = role_means[r_]
                if prev is not None and rm.any():
                    # a thin rank inherits the next-shallower role's prior, never the (much larger)
                    # position-group mean
                    P.loc[rm, f"m_{share}"] = role_means.get(r_, prev)
            # MoM needs binomial-scale counts: use unweighted totals, not decay-weighted sums
            rc, rn = {"car": ("carries", "designed_runs"), "tgt": ("targets", "attempts")}[share]
            rr = raw.reindex(P.index)
            sel = gmask & (rr[rn] > 0) & P["on_roster"]
            k = _eb_strength(rr.loc[sel, rc], rr.loc[sel, rn], P.loc[sel, f"m_{share}"],
                             params.min_k, params.max_k) if sel.sum() >= 5 else params.max_k
            P.loc[gmask, f"k_{share}"] = k
            st.diagnostics["k"][f"{share}_{grp}"] = dict(
                k=round(k, 1), role_prior={r_: round(v, 4) for r_, v in role_means.items()},
                group_prior=round(m_grp, 4))
        P[f"a_{share}"] = P[c] + P[f"k_{share}"] * P[f"m_{share}"]
        P[f"b_{share}"] = (P[n] - P[c]).clip(lower=0) + P[f"k_{share}"] * (1 - P[f"m_{share}"])

    # catch rate by group
    P["a_cat"] = np.nan
    P["b_cat"] = np.nan
    for grp in ["QB", "RB", "WR", "TE", "OTH"]:
        ids = P.index[P["group"] == grp]
        r = raw.loc[ids]
        sel = r["targets"] > 0
        if sel.sum() == 0:
            continue
        m = float(np.clip(r.loc[sel, "catches"].sum() / r.loc[sel, "targets"].sum(), 0.05, 0.95))
        k = _eb_strength(r.loc[sel, "catches"], r.loc[sel, "targets"], m, params.min_k, params.max_k)
        P.loc[ids, "a_cat"] = P.loc[ids, "c_cat"] + k * m
        P.loc[ids, "b_cat"] = (P.loc[ids, "c_tgt"] - P.loc[ids, "c_cat"]).clip(lower=0) + k * (1 - m)
        st.diagnostics["k"][f"catch_{grp}"] = dict(prior_mean=round(m, 4), k=round(k, 1))

    # QB sack / scramble rates per dropback
    qbs = P.index[P["group"] == "QB"]
    r = raw.loc[qbs]
    for col_c, col_r, nm in [("c_sack", "sacks", "sack"), ("c_scr", "db_scrambles", "scr")]:
        sel = r["dropbacks"] > 0
        m = float(np.clip(r.loc[sel, col_r].sum() / max(r.loc[sel, "dropbacks"].sum(), 1), 0.005, 0.5))
        k = _eb_strength(r.loc[sel, col_r], r.loc[sel, "dropbacks"], m, params.min_k, params.max_k)
        P[f"a_{nm}"] = P[col_c] + k * m
        P[f"b_{nm}"] = (P["n_db"] - P[col_c]).clip(lower=0) + k * (1 - m)
        st.diagnostics["k"][f"{nm}_QB"] = dict(prior_mean=round(m, 4), k=round(k, 1))
    # Expected-TD rate per opportunity (carry / scramble / target), shrunk toward the
    # position-group rate. A goal-line back shows up here: few carries, high xTD per carry.
    st.league_td_rate = {}
    for kind, col in [("carry", "r_car"), ("scramble", "r_scr"), ("target", "r_tgt")]:
        P[col] = np.nan
        ok = tm_xtd[tm_xtd["kind"] == kind]
        st.league_td_rate[kind] = float(ok["p_td"].mean()) if len(ok) else 0.03
        for grp in ["QB", "RB", "WR", "TE", "OTH"]:
            ids = P.index[P["group"] == grp]
            d = ok[ok["player_id"].isin(ids)]
            if len(d) == 0:
                P.loc[ids, col] = st.league_td_rate[kind]
                continue
            m = float(d["p_td"].mean())
            k = _eb_mean_strength([g["p_td"].to_numpy(float) for _, g in d.groupby("player_id")],
                                  params.min_k, params.max_k)
            num = (d["p_td"] * d["w"]).groupby(d["player_id"]).sum().reindex(ids).fillna(0)
            cnt = d["w"].groupby(d["player_id"]).sum().reindex(ids).fillna(0)
            P.loc[ids, col] = ((num + k * m) / (cnt + k)).to_numpy()
            st.diagnostics["k"][f"xtd_{kind}_{grp}"] = dict(prior_mean=round(m, 4), k=round(k, 1))
    st.players = P

    # Game-to-game share volatility beyond binomial noise: a player's share in a given game
    # ~ Beta(psi*s, psi*(1-s)) around his long-run share s. Estimated by method of moments.
    st.diagnostics["psi"] = {}
    for share, c, n in [("car", "carries", "designed_runs"), ("tgt", "targets", "attempts")]:
        d = pg.copy()
        d["grp_tot"] = d.groupby(["game_id", "posteam", "group"])[c].transform("sum")
        d = d[d["player_id"].isin(on_roster) & (d["grp_tot"] > 0)].copy()
        n = "grp_tot"
        d["x"] = d[c] / d[n]
        for grp, dg in d.groupby("group"):
            g = dg.groupby("player_id")
            m = g["x"].transform("mean")
            dg = dg.assign(m=m, noise=m * (1 - m) / dg[n], dev2=(dg["x"] - m) ** 2)
            agg = dg.groupby("player_id").agg(games=("x", "size"), m=("m", "first"),
                                              dev2=("dev2", "sum"), noise=("noise", "mean"))
            agg = agg[(agg["games"] >= 4) & (agg["m"] >= 0.03)]
            if len(agg) < 5:
                st.psi[(share, grp)] = 1e4
                continue
            var_game = (agg["dev2"] / (agg["games"] - 1) - agg["noise"]).clip(lower=0)
            num = np.average(agg["m"] * (1 - agg["m"]), weights=agg["games"])
            den_ = np.average(var_game, weights=agg["games"])
            psi = float(np.clip(num / max(den_, 1e-9) - 1, 2, 1e4))
            st.psi[(share, grp)] = psi
            st.diagnostics["psi"][f"{share}_{grp}"] = round(psi, 1)

    # Team game-efficiency factor: within-team, game-level variation in yards per play
    # that iid per-play sampling cannot produce (defense, conditions, script).
    st.diagnostics["eff_sigma"] = {}
    eff = op[op["kind"].isin(["carry", "scramble"]) | ((op["kind"] == "target") & (op["caught"] == 1))].copy()
    eff["pool"] = np.where(eff["kind"] == "target", "catch", "rush")
    for pool, d in eff.groupby("pool"):
        tgm = d.groupby(["posteam", "game_id"])["yards"].agg(["mean", "size", "var"]).reset_index()
        tgm = tgm[tgm["size"] >= 5]
        # pooled WITHIN team-game per-play variance (league-pooled variance contains between-team spread)
        per_play_var = float(np.average(tgm["var"], weights=tgm["size"] - 1))
        tgm["team_mean"] = tgm.groupby("posteam")["mean"].transform("mean")
        G = tgm.groupby("posteam")["mean"].transform("size")
        dev2 = (tgm["mean"] - tgm["team_mean"]) ** 2 * G / (G - 1).clip(lower=1)
        var_obs = float(np.average(dev2, weights=tgm["size"]))
        var_iid = float(np.average(per_play_var / tgm["size"], weights=tgm["size"]))
        var_game = max(var_obs - var_iid, 0.0)
        sig = float(np.sqrt(var_game) / d["yards"].mean())
        st.eff_sigma[pool] = sig
        st.diagnostics["eff_sigma"][pool] = round(sig, 4)

    # Efficiency vs market-implied points. Estimated WITHIN team-season (team-season fixed
    # effects) so the coefficient is the matchup-driven part: a player's own yardage pool
    # already reflects his team's usual environment. Wind is a control because it is applied
    # separately. Pooled 2024-26 estimate ~ +1.5%/pt for yds/carry and yds/catch, t ~ 3.
    st.league_implied = float(np.average(tg["implied"], weights=tg["w"]))
    norm = tg.groupby("posteam").apply(lambda d: np.average(d["implied"], weights=d["w"]), include_groups=False)
    st.team_implied_norm = {t: float(v) for t, v in norm.items()}
    st.diagnostics["eff_implied"] = {}
    tgi = tg.set_index(["game_id", "posteam"])[["implied", "wind_eff", "season"]]
    for pool, d in eff.groupby("pool"):
        gm = d.groupby(["game_id", "posteam"])["yards"].agg(["mean", "size"]).join(tgi, how="inner").reset_index()
        gm = gm[(gm["size"] >= 5) & (gm["mean"] > 0)]
        if len(gm) < 50 or not params.implied_eff:
            st.eff_implied[pool] = 0.0
            continue
        use_pool = pool in params.implied_eff_pools
        gm["ts"] = gm["posteam"] + gm["season"].astype(str)
        gm["ly"] = np.log(gm["mean"])
        for col in ["ly", "implied", "wind_eff"]:
            gm[col + "_dm"] = gm[col] - gm.groupby("ts")[col].transform("mean")
        bx, sex, *_ = _wls(np.column_stack([gm["implied_dm"], gm["wind_eff_dm"]]), gm["ly_dm"].to_numpy(float),
                           gm["size"].to_numpy(float))
        slope = float(bx[0])
        # keep only a slope that the data distinguish from zero (|t| >= 2); otherwise 0
        st.eff_implied[pool] = slope if (use_pool and abs(slope) >= 2 * float(sex[0])) else 0.0
        st.diagnostics["eff_implied"][pool] = dict(per_point=round(slope, 4), se=round(float(sex[0]), 4),
                                                   used=round(st.eff_implied[pool], 4))

    # ---------------- team position-group shares ----------------
    for team, d in pg.groupby("posteam"):
        # one denominator per game (not per player row)
        g_den = d.drop_duplicates("game_id")
        for kind, c, n in [("car", "carries", "designed_runs"), ("tgt", "targets", "attempts")]:
            tot = float((g_den[n] * g_den["w"]).sum())
            shares = {}
            for grp, dg in d.groupby("group"):
                shares[grp] = float((dg[c] * dg["w"]).sum() / max(tot, 1e-9))
            st.team_group_share[(team, kind)] = shares

    # ---------------- yardage pools ----------------
    op["pool"] = op["kind"].map({"carry": "rush", "scramble": "rush", "target": "catch"})
    yo = op[op["pool"].notna()]
    yo = yo[(yo["pool"] == "rush") | (yo["caught"] == 1)]
    for (grp, pool), d in yo.groupby(["group", "pool"]):
        st.league_pools[(grp, pool)] = d["yards"].to_numpy(float)
        if params.league_recency:
            cw = np.cumsum(d["w"].to_numpy(float))
            st.league_pool_cdf[(grp, pool)] = cw / cw[-1]
        by_player = [g["yards"].to_numpy(float) for _, g in d.groupby("player_id")]
        st.k_yards[(grp, pool)] = _eb_mean_strength(by_player, params.min_k, params.max_k)
    st.diagnostics["k_yards"] = {f"{g}_{p}": round(k, 1) for (g, p), k in st.k_yards.items()}
    for (pid, pool), d in yo.groupby(["player_id", "pool"]):
        st.pools[(pid, pool)] = (d["yards"].to_numpy(float), d["w"].to_numpy(float))

    return st


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _league_draws(rng, league, n, cdf=None):
    """n draws from the league pool: uniform, or recency-weighted via its cumulative weights."""
    if cdf is None:
        return league[rng.integers(0, len(league), n)]
    return league[np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), len(league) - 1)]


def _bootstrap_sums(rng, counts, own, own_w, league, k, league_cdf=None):
    """Sum of `counts[s]` per-play yards for each sim s, drawn from a mixture:
    own plays (recency-weighted) with prob n_eff/(n_eff+k), else league pool."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(len(counts))
    seg = np.repeat(np.arange(len(counts)), counts)
    draws = _league_draws(rng, league, total, league_cdf)
    if own is not None and len(own) > 0:
        n_eff = float(own_w.sum())
        p_own = n_eff / (n_eff + k)
        use_own = rng.random(total) < p_own
        n_own = int(use_own.sum())
        if n_own:
            draws[use_own] = own[rng.choice(len(own), n_own, p=own_w / own_w.sum())]
    return np.bincount(seg, weights=draws, minlength=len(counts))


def _beta_draws(rng, a, b, n):
    a = np.maximum(np.asarray(a, float), 1e-3)
    b = np.maximum(np.asarray(b, float), 1e-3)
    return rng.beta(a[:, None], b[:, None], size=(len(a), n))


def _group_scaled_shares(rng, P: pd.DataFrame, kind: str, team_shares: dict, n: int,
                         psi: dict | None = None, present: np.ndarray | None = None):
    """Per-sim share draws, rescaled so each position group keeps its team-level share.

    Removing an OUT player therefore redistributes his volume within his position group.
    `present` (players x sims, bool) is False where a player sits in that sim (availability
    model): his volume moves to his position group in that sim only.
    Returns (shares[players, sims], other[sims]).
    """
    draws = _beta_draws(rng, P[f"a_{kind}"], P[f"b_{kind}"], n)   # long-run shares (posterior)
    groups = P["group"].to_numpy()
    out = np.zeros_like(draws)
    for grp in np.unique(groups):
        idx = groups == grp
        target = team_shares.get(grp, np.nan)
        if np.isnan(target) or target <= 0:
            out[idx] = draws[idx]
            continue
        within = draws[idx] / np.maximum(draws[idx].sum(axis=0), 1e-9)
        if psi and idx.sum() > 1:
            # This game's within-group split ~ Dirichlet(psi * long-run split): marginals are
            # Beta(psi*s, psi*(1-s)), the same form psi was estimated under, so the volatility
            # survives the group-total constraint (independent Betas + renormalizing did not).
            alpha = np.maximum(psi.get((kind, grp), 1e4) * within, 1e-3)
            g = rng.gamma(alpha)
            within = g / np.maximum(g.sum(axis=0), 1e-12)
        if present is not None:
            within = within * present[idx]
            within = within / np.maximum(within.sum(axis=0), 1e-12)
        out[idx] = within * target
    tot = out.sum(axis=0)
    scale = np.where(tot > 0.995, 0.995 / tot, 1.0)
    out = out * scale
    return out, 1.0 - out.sum(axis=0)


def own_samples(d: dict, key: str) -> np.ndarray | None:
    """A player's own stat samples, restricted to the sims in which he plays.

    His teammates' arrays already include the volume he leaves behind when he sits; his own
    props are priced conditional on playing (a sportsbook voids a prop on a scratched player).
    """
    x = d.get(key)
    if x is None:
        return None
    m = d.get("played")
    if m is None or m.all() or not m.any():
        return x
    return x[m]


@dataclass
class GameInputs:
    team: str
    opponent: str
    implied: float
    team_spread: float
    wind_mph: float = 0.0       # 0 for dome/closed roof
    outs: tuple = ()            # player_ids ruled out
    qb: str | None = None       # starting QB player_id override
    adds: tuple = ()            # players returning from injury (not on the recent-snaps roster)
    avail: dict = field(default_factory=dict)   # player_id -> P(plays); missing = 1 (QBs always 1)


def simulate_team(st: State, gi: GameInputs, n: int = 10000, seed: int = 7) -> dict:
    """Simulate one offense. Returns {player_id: {stat: np.ndarray(n)}} plus 'meta'."""
    rng = np.random.default_rng(seed)
    P = st.players
    roster = [p for p in list(st.roster.get(gi.team, [])) + [a for a in gi.adds if a in P.index]
              if p not in set(gi.outs)]
    roster = list(dict.fromkeys(roster))
    R = P.loc[roster].copy()
    if R.empty:
        return {"meta": {"error": f"no active roster for {gi.team}"}}

    # starting QB: override, else most weighted dropbacks among active QBs
    qbs = R[R["group"] == "QB"]
    qb_id = gi.qb if gi.qb in R.index else (qbs["n_db"].idxmax() if len(qbs) else None)
    # backup QBs keep designed-run share but no dropbacks
    # ---- team volume
    plays_mu = (st.plays_beta[0] + st.plays_beta[1] * gi.implied + st.plays_beta[2] * gi.team_spread
                + st.team_plays_eff.get(gi.team, 0.0))
    plays = np.clip(np.rint(rng.normal(plays_mu, st.plays_sigma, n)), 35, 100).astype(int)
    logit = (st.pass_beta[0] + st.pass_beta[1] * gi.team_spread + st.pass_beta[2] * gi.wind_mph
             + st.team_pass_eff.get(gi.team, 0.0) + rng.normal(0, st.pass_tau, n))
    q = 1 / (1 + np.exp(-logit))
    dropbacks = rng.binomial(plays, q)
    runs = plays - dropbacks

    if qb_id is not None:
        qrow = P.loc[qb_id]
        p_sack = rng.beta(max(qrow["a_sack"], 1e-3), max(qrow["b_sack"], 1e-3), n)
        p_scr = rng.beta(max(qrow["a_scr"], 1e-3), max(qrow["b_scr"], 1e-3), n)
    else:
        p_sack = np.full(n, 0.065)
        p_scr = np.full(n, 0.05)
    p_att = np.clip(1 - p_sack - p_scr, 0.5, 1)
    pv = np.column_stack([p_att, p_sack, 1 - p_att - p_sack]).clip(0, 1)
    pv = pv / pv.sum(axis=1, keepdims=True)
    db_out = rng.multinomial(dropbacks, pv)
    attempts, scrambles = db_out[:, 0], db_out[:, 2]

    wind_factor = 1.0 + st.wind_ypa_slope * gi.wind_mph / st.league_ypa
    sr, sc = st.eff_sigma.get("rush", 0.0), st.eff_sigma.get("catch", 0.0)
    # this matchup vs the team's usual market environment
    d_imp = float(np.clip(gi.implied - st.team_implied_norm.get(gi.team, st.league_implied),
                          -st.params.eff_clip, st.params.eff_clip))
    m_rush = float(np.exp(st.eff_implied.get("rush", 0.0) * d_imp))
    m_rec = float(np.exp(st.eff_implied.get("catch", 0.0) * d_imp))
    f_rush = (rng.lognormal(-sr ** 2 / 2, sr, n) if sr > 0 else np.ones(n)) * m_rush
    f_rec = (rng.lognormal(-sc ** 2 / 2, sc, n) if sc > 0 else np.ones(n)) * wind_factor * m_rec

    # ---- availability: each non-QB plays in a sim with his estimated probability (default 1)
    p_play = np.array([1.0 if (pid == qb_id or R.loc[pid, "group"] == "QB")
                       else float(np.clip(gi.avail.get(pid, 1.0), 0.0, 1.0)) for pid in R.index])
    present = rng.random((len(R), n)) < p_play[:, None] if (p_play < 1).any() else None

    # ---- allocation
    R_car, other_car = _group_scaled_shares(rng, R, "car", st.team_group_share.get((gi.team, "car"), {}), n,
                                            st.psi, present)
    R_tgt, other_tgt = _group_scaled_shares(rng, R, "tgt", st.team_group_share.get((gi.team, "tgt"), {}), n,
                                            st.psi, present)

    carries = rng.multinomial(runs, np.vstack([R_car, other_car]).T)[:, :-1].T
    targets = rng.multinomial(attempts, np.vstack([R_tgt, other_tgt]).T)[:, :-1].T
    catch_p = _beta_draws(rng, R["a_cat"].fillna(1), R["b_cat"].fillna(1), n)
    catches = rng.binomial(targets, catch_p)

    # TDs: each player's intensity = simulated volume x his xTD-per-opportunity rate,
    # then all intensities are scaled so the team's expected TDs match the market
    # (lambda from implied points). Volume and TDs are therefore correlated by construction.
    lam = max(st.td_beta[0] + st.td_beta[1] * gi.implied, 0.05)
    is_qb = (R.index.to_numpy() == qb_id)[:, None]
    mu = (carries * R["r_car"].to_numpy()[:, None] + targets * R["r_tgt"].to_numpy()[:, None]
          + np.where(is_qb, scrambles[None, :], 0) * R["r_scr"].to_numpy()[:, None])
    other_mu = ((runs - carries.sum(axis=0)) * st.league_td_rate["carry"]
                + (attempts - targets.sum(axis=0)) * st.league_td_rate["target"])
    kappa = lam / max(float((mu.sum(axis=0) + other_mu).mean()), 1e-6)
    tds = rng.poisson(kappa * mu)

    out = {}
    total_rec_yds = np.zeros(n)
    for i, pid in enumerate(R.index):
        grp = R.loc[pid, "group"]
        kr = (grp, "rush") if (grp, "rush") in st.league_pools else ("RB", "rush")
        kc = (grp, "catch") if (grp, "catch") in st.league_pools else ("WR", "catch")
        own_r = st.pools.get((pid, "rush"), (None, None))
        own_c = st.pools.get((pid, "catch"), (None, None))
        rush_att = carries[i] + (scrambles if pid == qb_id else 0)
        rush_yds = _bootstrap_sums(rng, rush_att, own_r[0], own_r[1], st.league_pools[kr],
                                   st.k_yards.get((grp, "rush"), 200), st.league_pool_cdf.get(kr)) * f_rush
        rec_yds = _bootstrap_sums(rng, catches[i], own_c[0], own_c[1], st.league_pools[kc],
                                  st.k_yards.get((grp, "catch"), 200), st.league_pool_cdf.get(kc)) * f_rec
        total_rec_yds += rec_yds
        out[pid] = dict(rush_att=rush_att, rush_yds=rush_yds, targets=targets[i],
                        receptions=catches[i], rec_yds=rec_yds, tds=tds[i],
                        played=present[i] if present is not None else np.ones(n, bool),
                        p_play=float(p_play[i]))

    # passing yards = all receiving yards, incl. targets to players outside the modeled roster
    other_tg = attempts - targets.sum(axis=0)
    # attempts without a receiver (throwaways, spikes) complete 0%; the rest go to unmodeled players
    no_recv = np.minimum(rng.binomial(attempts, st.no_receiver_rate), other_tg)
    other_catch = rng.binomial(other_tg - no_recv, st.league_catch)
    other_yds = _bootstrap_sums(rng, other_catch, None, None, st.league_pools[("WR", "catch")], 1,
                                st.league_pool_cdf.get(("WR", "catch"))) * f_rec
    if qb_id is not None:
        out[qb_id]["pass_att"] = attempts
        out[qb_id]["pass_yds"] = total_rec_yds + other_yds
    # yards are integers in box scores: round so 'X+' thresholds mean the same thing as the contract
    for pid in list(out):
        for key in ("rush_yds", "rec_yds", "pass_yds"):
            if key in out[pid]:
                out[pid][key] = np.rint(out[pid][key])
    out["meta"] = dict(team=gi.team, qb=qb_id, plays_mu=float(plays_mu),
                       pass_share_mean=float(q.mean()), td_lambda=float(lam), td_kappa=float(kappa),
                       wind_factor=float(wind_factor), roster=list(R.index),
                       implied_vs_norm=d_imp, eff_mult_rush=m_rush, eff_mult_rec=m_rec)
    return out

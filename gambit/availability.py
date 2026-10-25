"""
P(plays) for every simulated skill player who is not ruled out before kickoff.

Before kickoff the app knows the Friday injury report, the weekly roster status (IR, PUP,
suspended...) and each player's recent snaps. It does not know the game-day inactive list.
Treating everyone else as certain to play was wrong in two places (2025 + 2026 Weeks 1-6,
non-QB players on the simulated roster, using only pre-kickoff information):

  * listed Questionable: took a snap 66% / 63% of the time;
  * not listed, but did not play the team's previous game (healthy scratch, coming back, a
    depth player who rotated out): 37% / 36%. This group carried more lost target share than
    the Questionable group.
  * not listed and played the previous game: 94% / 92%.

Model: two logistic regressions. Questionable players: final practice status (Full /
Limited / DNP) and recent role (mean offensive snap share over the team's last 3 games, 0 for
games he missed). Everyone else: practice status if he was on the practice report, whether he
played the team's previous game, role, and role x previous game. Both are fit on every
team-week before the cutoff (weeks 4+ so the 3-game window stays in one
season). Out / Doubtful players and anyone on reserve, suspended, exempt, released or traded
stay OUT and are not modeled: none of them took a snap. QBs are not modeled: the starter is
chosen on the Slate tab.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SKILL = {"RB", "WR", "TE", "FB"}
PRACTICE = {"Full Participation in Practice": "Full",
            "Limited Participation in Practice": "Limited",
            "Did Not Participate In Practice": "DNP"}
Q_FEATURES = ["const", "limited", "dnp", "role"]
U_FEATURES = ["const", "limited", "dnp", "played_prev", "role", "role_x_prev"]


def _t(season, week):
    return season * 20 + week


def window_features(snaps: pd.DataFrame, team: str, t0: int, n: int = 3) -> pd.DataFrame:
    """Players with an offensive snap for `team` in its last n games before t0.

    Returns player_id-indexed frame: role (mean snap share over those n team games, 0 when he
    did not play), played_prev (snap in the team's most recent game).
    """
    d = snaps[(snaps["posteam"] == team)]
    tt = d["season"] * 20 + d["week"]
    games = np.sort(tt[tt < t0].unique())[-n:]
    if len(games) == 0:
        return pd.DataFrame(columns=["role", "played_prev"])
    w = d[tt.isin(games)]
    wt = w["season"] * 20 + w["week"]
    role = w.groupby("player_id")["offense_pct"].sum() / len(games)
    prev = set(w.loc[wt == games[-1], "player_id"])
    out = pd.DataFrame({"role": role.clip(0, 1)})
    out["played_prev"] = out.index.isin(prev)
    return out


def _parts(df: pd.DataFrame):
    q = (df["report_status"] == "Questionable").to_numpy()
    prac = df["practice_status"].map(PRACTICE)
    lim, dnp = (prac == "Limited").to_numpy(float), (prac == "DNP").to_numpy(float)
    role = df["role"].fillna(0.0).clip(0, 1).to_numpy(float)
    prev = df["played_prev"].astype(float).to_numpy()
    one = np.ones(len(df))
    Xq = np.column_stack([one, lim, dnp, role])
    Xu = np.column_stack([one, lim, dnp, prev, role, role * prev])
    return q, Xq, Xu


def _logit(X, y, l2):
    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(b[1:] ** 2))

    def grad(b):
        g = X.T @ (1 / (1 + np.exp(-(X @ b))) - y)
        g[1:] += l2 * b[1:]
        return g

    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


@dataclass
class Availability:
    """Two logistic regressions: Questionable players, and everyone else not ruled out.

    One pooled model was tried first and was worse than a constant rate on Questionable
    players out of sample (Brier 0.238 vs 0.224, 2025 + 2026): its snap-share slope is set by
    unlisted depth players and over-states how often a Questionable starter plays. Separate
    fits: Questionable 0.208.
    """
    beta_q: np.ndarray = field(default_factory=lambda: np.array([0.66, 0, 0, 0]))
    beta_u: np.ndarray = field(default_factory=lambda: np.array([6.0, 0, 0, 0, 0, 0]))
    n: int = 0
    diagnostics: dict = field(default_factory=dict)

    def prob(self, df: pd.DataFrame) -> np.ndarray:
        q, Xq, Xu = _parts(df)
        z = np.where(q, Xq @ self.beta_q, Xu @ self.beta_u)
        return 1 / (1 + np.exp(-z))


def week_frame(snaps: pd.DataFrame, inj_week: pd.DataFrame, team: str, season: int, week: int,
               players: list | None = None) -> pd.DataFrame:
    """Features for one team-week. `players` defaults to the snap window's players.

    inj_week: that week's injury rows (gsis_id, report_status, practice_status).
    """
    wf = window_features(snaps, team, _t(season, week))
    if players is not None:
        wf = wf.reindex(players)
        wf["role"] = wf["role"].fillna(0.0)
        wf["played_prev"] = wf["played_prev"].fillna(False).astype(bool)
    rep = inj_week.drop_duplicates("gsis_id").set_index("gsis_id")
    wf["report_status"] = wf.index.map(rep["report_status"]) if len(rep) else None
    wf["practice_status"] = wf.index.map(rep["practice_status"]) if len(rep) else None
    return wf


def training_rows(inj_hist: pd.DataFrame, snaps: pd.DataFrame, positions: pd.Series,
                  unavailable: dict, cutoff: tuple[int, int] | None = None, min_week: int = 4) -> pd.DataFrame:
    """Every (team, week, candidate) before the cutoff, with y = took an offensive snap for that team.

    unavailable: (season, week) -> set of gsis_ids ruled out pre-kickoff by roster status.
    """
    t_cut = _t(*cutoff) if cutoff else 10 ** 9
    sn = snaps.assign(t=snaps["season"] * 20 + snaps["week"])
    skill = set(positions[positions.isin(SKILL)].index)
    sn = sn[sn["player_id"].isin(skill)]
    rows = []
    for (season, week), wk in sn[(sn["t"] < t_cut) & (sn["week"] >= min_week)].groupby(["season", "week"]):
        inj_w = inj_hist[(inj_hist["season"] == season) & (inj_hist["week"] == week)]
        out = set(inj_w.loc[inj_w["report_status"].isin(["Out", "Doubtful"]), "gsis_id"])
        out |= unavailable.get((season, week), set())
        for team, g in wk.groupby("posteam"):
            f = week_frame(sn, inj_w, team, season, week)
            f = f[~f.index.isin(out)]
            if f.empty:
                continue
            f = f.assign(season=season, week=week, t=_t(season, week), team=team,
                         y=f.index.isin(set(g["player_id"])))
            rows.append(f)
    return pd.concat(rows) if rows else pd.DataFrame()


def fit(train: pd.DataFrame, l2: float = 1.0) -> Availability:
    if train.empty:
        return Availability()
    q, Xq, Xu = _parts(train)
    y = train["y"].to_numpy(float)
    a = Availability(n=len(train))
    if q.sum() >= 30:
        a.beta_q = _logit(Xq[q], y[q], l2)
    if (~q).sum() >= 30:
        a.beta_u = _logit(Xu[~q], y[~q], l2)
    p = a.prob(train)
    miss = ~q & ~train["played_prev"].to_numpy(bool)
    a.diagnostics = dict(
        n=len(train), play_rate=round(float(y.mean()), 3),
        coef_questionable=dict(zip(Q_FEATURES, np.round(a.beta_q, 3).tolist())),
        coef_other=dict(zip(U_FEATURES, np.round(a.beta_u, 3).tolist())),
        questionable=dict(n=int(q.sum()), obs=round(float(y[q].mean()), 3) if q.any() else None,
                          pred=round(float(p[q].mean()), 3) if q.any() else None),
        unlisted_missed_prev=dict(n=int(miss.sum()), obs=round(float(y[miss].mean()), 3) if miss.any() else None))
    return a


def pregame_unavailable(rstat_by_season: dict, unavailable_fn) -> dict:
    """(season, week) -> ids ruled out before kickoff by roster status (game-day INA excluded)."""
    unav = {}
    for season, rs in rstat_by_season.items():
        if rs is None or rs.empty:
            continue
        for week in sorted(rs["week"].unique()):
            unav[(season, int(week))] = set(unavailable_fn(rs, int(week), pregame=True))
    return unav


def fit_before(train_all: pd.DataFrame, cutoff: tuple[int, int]) -> Availability:
    """Rows only use information before their own week, so slicing a full table is walk-forward safe."""
    if train_all.empty:
        return Availability()
    return fit(train_all[train_all["t"] < _t(*cutoff)])

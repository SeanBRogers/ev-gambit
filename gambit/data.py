"""
Data layer. One source: nflverse via nflreadpy (free, no key).

Everything the model needs is derived from regular-season play-by-play:
  - team_games:   one row per team per game (volume, lines, environment)
  - opportunities: one row per player opportunity (carry, scramble, target, dropback)
  - league xTD table: P(TD | opportunity type, yardline bucket)

Conventions (verified against nflverse data, Oct 2026):
  - spread_line > 0 means the HOME team is favored (mean home margin +6.4 when > 0).
  - QB scrambles: play_type == 'run', pass == 1, rush == 0, rusher_player_id set.
  - Designed runs: rush == 1 and play_type == 'run' (kneels excluded from volume).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("GAMBIT_CACHE", Path.home() / ".cache" / "gambit"))

PBP_COLS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "play_type", "pass", "rush", "qb_dropback", "qb_scramble",
    "sack", "complete_pass", "incomplete_pass", "interception", "two_point_attempt",
    "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name", "td_player_id",
    "rush_touchdown", "pass_touchdown", "yardline_100",
    "passing_yards", "rushing_yards", "receiving_yards",
    "spread_line", "total_line", "roof", "wind", "temp",
]

YARD_BUCKETS = [0, 1, 2, 3, 5, 10, 20, 100]  # yardline_100 bucket edges (right-inclusive)


def _cached(name: str, fetch, max_age_hours: float | None = 12.0, retries: int = 2) -> pd.DataFrame:
    """Parquet cache with age-based refresh and retry; falls back to a stale copy on network failure."""
    import time

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    fresh = path.exists() and (max_age_hours is None or
                               (time.time() - path.stat().st_mtime) < max_age_hours * 3600)
    if fresh:
        return pd.read_parquet(path)
    err = None
    for attempt in range(retries + 1):
        try:
            df = fetch()
            df.to_parquet(path, index=False)
            return df
        except Exception as e:  # transient network errors are common on release assets
            err = e
            time.sleep(2 * (attempt + 1))
    if path.exists():
        return pd.read_parquet(path)
    raise err


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pbp(seasons: list[int], refresh_current: bool = True) -> pd.DataFrame:
    """Regular-season pbp for `seasons`, cached as parquet.

    Completed seasons are cached permanently; the current season is re-pulled
    when refresh_current is True.
    """
    import nflreadpy as nfl

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current = max(seasons)
    frames = []
    for s in seasons:
        path = CACHE_DIR / f"pbp_{s}.parquet"
        if path.exists() and not (refresh_current and s == current):
            frames.append(pd.read_parquet(path))
            continue
        df = nfl.load_pbp([s]).select([c for c in PBP_COLS]).to_pandas()
        df = df[df["season_type"] == "REG"].copy()
        df.to_parquet(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_schedule(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    cols = ["game_id", "season", "week", "game_type", "gameday", "gametime",
            "away_team", "home_team", "away_score", "home_score", "result",
            "spread_line", "total_line", "roof", "stadium", "stadium_id",
            "away_rest", "home_rest", "temp", "wind"]
    return _cached(f"schedule_{min(seasons)}_{max(seasons)}",
                   lambda: nfl.load_schedules(seasons).select(cols).to_pandas(), max_age_hours=3)


def load_positions(seasons: list[int]) -> pd.DataFrame:
    """player_id → most recent position (gsis ids match pbp ids)."""
    import nflreadpy as nfl

    ps = _cached(f"positions_{min(seasons)}_{max(seasons)}", lambda: nfl.load_player_stats(seasons).select(
        ["player_id", "player_display_name", "position", "season", "week", "team"]).to_pandas())
    ps = ps.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    return ps.rename(columns={"player_display_name": "display_name"})[
        ["player_id", "display_name", "position", "team"]]


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    """Offensive snaps per player-game, keyed by gsis id (PFR ids mapped via nflverse players)."""
    import nflreadpy as nfl

    sc = _cached(f"snaps_{min(seasons)}_{max(seasons)}", lambda: nfl.load_snap_counts(seasons).select(
        ["game_id", "season", "week", "game_type", "pfr_player_id", "team", "offense_snaps", "offense_pct"]
    ).to_pandas())
    sc = sc[sc["game_type"] == "REG"]
    ids = _cached("player_ids", lambda: nfl.load_players().select(["gsis_id", "pfr_id"]).to_pandas().dropna(),
                  max_age_hours=72)
    sc = sc.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sc = sc[sc["offense_snaps"].fillna(0) > 0]
    return sc.rename(columns={"gsis_id": "player_id", "team": "posteam"})[
        ["player_id", "game_id", "season", "week", "posteam", "offense_snaps", "offense_pct"]]


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    """Official weekly box scores — used only to grade predictions."""
    import nflreadpy as nfl

    cols = ["player_id", "player_display_name", "position", "season", "week", "season_type",
            "team", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
            "receiving_yards", "receiving_tds", "attempts", "passing_yards"]
    df = _cached(f"player_stats_{min(seasons)}_{max(seasons)}",
                 lambda: nfl.load_player_stats(seasons).select(cols).to_pandas())
    return df[df["season_type"] == "REG"].drop(columns="season_type")


def load_injuries(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    try:
        df = _cached(f"injuries_{season}", lambda: nfl.load_injuries([season]).to_pandas(), max_age_hours=3)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "position",
                                     "report_status", "practice_status"])
    if "game_type" in df:
        df = df[df["game_type"] == "REG"]
    return df[["season", "week", "team", "gsis_id", "full_name", "position", "report_status", "practice_status"]]


def load_injuries_hist(seasons: list[int]) -> pd.DataFrame:
    """Injury reports for several seasons (regular season), for fitting P(plays | Questionable)."""
    frames = [load_injuries(s) for s in seasons]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else frames[0]
    return df.dropna(subset=["gsis_id"])


# Weekly roster status codes that mean "cannot play this week". The weekly injury report
# never lists players on injured reserve/PUP/NFI, suspended or exempt, so the report alone
# leaves them in the simulated roster with their pre-injury share (e.g. KC's Tyquan Thornton,
# IR since Week 4, still drew 3.4 targets in the Week 7 export).
#   RES = reserve (IR, PUP, NFI, suspended), EXE = commissioner exempt, INA = game-day
#   inactive (posted ~90 min pre-kickoff), CUT/RET = released/retired, TRD/TRT = traded.
UNAVAILABLE_STATUS = {"RES", "EXE", "INA", "CUT", "RET", "TRD", "TRT"}


def load_roster_status(season: int) -> pd.DataFrame:
    """Weekly roster status per player (gsis id). Empty frame on failure."""
    import nflreadpy as nfl

    try:
        df = _cached(f"roster_status_{season}", lambda: nfl.load_rosters_weekly([season]).select(
            ["season", "week", "team", "gsis_id", "full_name", "status", "status_description_abbr"]
        ).to_pandas(), max_age_hours=3)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "status"])
    return df.dropna(subset=["gsis_id"])


def unavailable_ids(roster_status: pd.DataFrame, week: int, pregame: bool = False) -> dict:
    """gsis_id -> status for players who cannot play in `week`.

    Uses that week's roster snapshot; if it isn't published yet, the latest earlier week
    (a player on IR last week is still on IR unless designated to return — the user can
    add him back on the Slate tab).

    pregame=True drops `week`'s own INA rows. INA is the game-day inactive list: nflverse
    shows none for games not yet played (checked Week 7 2026: 0 INA for the 10 unplayed
    teams, 6-7 per finished team), so a backtest that reads the target week's INA is using
    information published ~90 minutes before kickoff, after the injury report.
    """
    rs = roster_status[roster_status["week"] <= week]
    if pregame:
        rs = rs[~((rs["week"] == week) & (rs["status"] == "INA"))]
    if rs.empty:
        return {}
    last_wk = rs.groupby("gsis_id")["week"].transform("max")
    latest = rs[rs["week"] == last_wk]
    if pregame:
        # his target-week row was INA (game-day): he was on the active roster before kickoff,
        # so an earlier week's status must not stand in for it
        ina_now = set(roster_status.loc[(roster_status["week"] == week) & (roster_status["status"] == "INA"), "gsis_id"])
        latest = latest[~latest["gsis_id"].isin(ina_now)]
    # a traded player can have two rows that week (TRD on the old team, ACT on the new one)
    active = set(latest.loc[latest["status"] == "ACT", "gsis_id"])
    bad = latest[latest["status"].isin(UNAVAILABLE_STATUS) & ~latest["gsis_id"].isin(active)]
    return dict(zip(bad["gsis_id"], bad["status"]))


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------

def implied_points(spread_line: float, total_line: float, is_home: bool) -> float:
    """Market-implied team points. spread_line > 0 = home favored."""
    s = spread_line if is_home else -spread_line
    return (total_line + s) / 2.0


def _clean(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["posteam"].notna() & (pbp["two_point_attempt"].fillna(0) == 0)].copy()
    for c in ["pass", "rush", "qb_dropback", "qb_scramble", "sack", "complete_pass",
              "incomplete_pass", "interception", "rush_touchdown", "pass_touchdown"]:
        p[c] = p[c].fillna(0).astype(int)
    p["designed_run"] = ((p["rush"] == 1) & (p["play_type"] == "run")).astype(int)
    p["attempt"] = ((p["complete_pass"] + p["incomplete_pass"] + p["interception"]) > 0).astype(int)
    # official attempts exclude sacks and scrambles; dropbacks include both
    p["dropback"] = (p["qb_dropback"] == 1).astype(int)
    p["is_home"] = p["posteam"] == p["home_team"]
    p["outdoor"] = p["roof"].isin(["outdoors", "open"])
    return p


def build_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense): volume, market lines, environment."""
    p = _clean(pbp)
    g = p.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(
        designed_runs=("designed_run", "sum"),
        dropbacks=("dropback", "sum"),
        attempts=("attempt", "sum"),
        sacks=("sack", "sum"),
        scrambles=("qb_scramble", "sum"),
        pass_yards=("passing_yards", "sum"),
        rush_tds=("rush_touchdown", "sum"),
        pass_tds=("pass_touchdown", "sum"),
        is_home=("is_home", "first"),
        spread_line=("spread_line", "first"),
        total_line=("total_line", "first"),
        outdoor=("outdoor", "first"),
        wind=("wind", "first"),
        temp=("temp", "first"),
    )
    g["plays"] = g["designed_runs"] + g["dropbacks"]
    g["off_tds"] = g["rush_tds"] + g["pass_tds"]
    g["implied"] = [implied_points(s, t, h) for s, t, h in
                    zip(g["spread_line"], g["total_line"], g["is_home"])]
    g["team_spread"] = np.where(g["is_home"], g["spread_line"], -g["spread_line"])  # + = favored
    g["wind_eff"] = np.where(g["outdoor"], g["wind"].fillna(0), 0.0)
    g = g[g["plays"] > 20]  # drop truncated/garbage rows
    return g.reset_index(drop=True)


def _yard_bucket(y: pd.Series) -> pd.Series:
    return pd.cut(y.fillna(75), bins=YARD_BUCKETS, labels=False, include_lowest=True)


def build_opportunities(pbp: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per player opportunity.

    kind ∈ {carry, scramble, target, dropback}
      carry:    designed run, player = rusher, yards = rushing_yards
      scramble: QB scramble, player = rusher (QB), yards = rushing_yards
      target:   pass attempt with a receiver; caught flag; yards = receiving_yards (0 if incomplete)
      dropback: per passer, outcome ∈ {attempt, sack, scramble}; yards = passing_yards on attempts
    td = 1 if this player scored on this play (rush or receiving TD).
    """
    p = _clean(pbp)
    p["ybucket"] = _yard_bucket(p["yardline_100"])
    base = ["game_id", "season", "week", "posteam", "ybucket"]

    carries = p[(p["designed_run"] == 1) & p["rusher_player_id"].notna()][base + ["rusher_player_id", "rusher_player_name", "rushing_yards", "rush_touchdown"]]
    carries = carries.rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "name",
                                      "rushing_yards": "yards", "rush_touchdown": "td"})
    carries["kind"] = "carry"

    scr = p[(p["qb_scramble"] == 1) & p["rusher_player_id"].notna()][base + ["rusher_player_id", "rusher_player_name", "rushing_yards", "rush_touchdown"]]
    scr = scr.rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "name",
                              "rushing_yards": "yards", "rush_touchdown": "td"})
    scr["kind"] = "scramble"

    tg = p[(p["attempt"] == 1) & p["receiver_player_id"].notna()][base + ["receiver_player_id", "receiver_player_name", "receiving_yards", "complete_pass", "pass_touchdown"]]
    tg = tg.rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "name",
                            "receiving_yards": "yards", "complete_pass": "caught", "pass_touchdown": "td"})
    tg["kind"] = "target"
    tg["yards"] = tg["yards"].fillna(0.0)

    # Scramble rows may carry the QB only as rusher; coalesce so the QB owns every dropback.
    p["qb_id"] = p["passer_player_id"].where(p["passer_player_id"].notna(),
                                             p["rusher_player_id"].where(p["qb_scramble"] == 1))
    p["qb_name"] = p["passer_player_name"].where(p["passer_player_name"].notna(),
                                                 p["rusher_player_name"].where(p["qb_scramble"] == 1))
    db = p[(p["dropback"] == 1) & p["qb_id"].notna()][base + ["qb_id", "qb_name", "passing_yards", "attempt", "sack", "qb_scramble"]]
    db = db.rename(columns={"qb_id": "player_id", "qb_name": "name", "passing_yards": "yards"})
    db["kind"] = "dropback"
    db["yards"] = np.where(db["attempt"] == 1, db["yards"].fillna(0.0), 0.0)
    db["td"] = 0

    out = pd.concat([carries, scr, tg, db], ignore_index=True)
    for c in ["caught", "attempt", "sack", "qb_scramble"]:
        if c in out:
            out[c] = out[c].fillna(0).astype(int)
    out["yards"] = out["yards"].fillna(0.0).astype(float)
    out["td"] = out["td"].fillna(0).astype(int)
    return out


def league_xtd(opps: pd.DataFrame) -> pd.DataFrame:
    """P(TD | kind, yardline bucket) for carries, scrambles and targets."""
    o = opps[opps["kind"].isin(["carry", "scramble", "target"])]
    t = o.groupby(["kind", "ybucket"], as_index=False)["td"].agg(["sum", "count"]).reset_index()
    t["p_td"] = (t["sum"] + 0.5) / (t["count"] + 1.0)
    return t[["kind", "ybucket", "p_td"]]


def game_order(team_games: pd.DataFrame) -> pd.DataFrame:
    """Chronological index of each team's games (for recency decay)."""
    tg = team_games[["game_id", "season", "week", "posteam"]].drop_duplicates()
    tg = tg.sort_values(["posteam", "season", "week"])
    tg["team_game_idx"] = tg.groupby("posteam").cumcount()
    return tg

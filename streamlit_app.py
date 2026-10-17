"""
EV+ GAMBIT v3 — NFL player-prop pricer for DraftKings Predictions.

Lean by design: no paid odds feed. Market lines, play-by-play, snaps and injuries come
from nflverse; weather from Open-Meteo. You type the DK contract price; the app returns
the model probability and the most you should pay after DK's per-contract fee.
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import streamlit as st

from gambit import calibrate as C
from gambit import data as D
from gambit import model as M
from gambit import price as PR
from gambit import weather as W

st.set_page_config(page_title="EV+ Gambit", page_icon="🏈", layout="wide")

MARKETS = {
    "Rushing yards": ("rush_yds", "rush_yds"),
    "Anytime TD": ("tds", "anytime_td"),
    "Passing yards": ("pass_yds", "pass_yds"),
    "Receptions": ("receptions", "receptions"),
    "Receiving yards": ("rec_yds", "rec_yds"),
}


def current_season(today: dt.date) -> int:
    return today.year if today.month >= 8 else today.year - 1


SEASON = current_season(dt.date.today())
SEASONS = [SEASON - 2, SEASON - 1, SEASON]


# ---------------------------------------------------------------------------
# Cached data and fitted state
# ---------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner="Loading nflverse data…")
def load_all(season: int):
    seasons = [season - 2, season - 1, season]
    pbp = D.load_pbp(seasons)
    pos = D.load_positions(seasons)
    snaps = D.load_snaps(seasons)
    sch = D.load_schedule([season])
    inj = D.load_injuries(season)
    return pbp, pos, snaps, sch, inj


@st.cache_resource(ttl=6 * 3600, show_spinner="Fitting model (≈10s)…")
def get_state(season: int, week: int):
    pbp, pos, snaps, _, _ = load_all(season)
    return M.fit_state(pbp, pos, (season, week), snaps=snaps)


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(home, stadium_id, gameday, gametime, home_stadium_id):
    return W.kickoff_weather(home, stadium_id, gameday, gametime, home_stadium_id)


def summarize(stv: M.State, sims: dict, cal: dict, use_cal: bool) -> pd.DataFrame:
    rows = []
    for pid, d in sims.items():
        if pid == "meta":
            continue
        p_td = float(np.mean(d["tds"] >= 1))
        if use_cal:
            p_td = float(C.apply(p_td, "anytime_td", cal))

        def q(x, lo=10, hi=90):
            return f"{np.median(x):.0f} ({np.percentile(x, lo):.0f}–{np.percentile(x, hi):.0f})"

        rows.append({
            "Player": stv.players.loc[pid, "name"], "Pos": stv.players.loc[pid, "position"],
            "Carries": round(float(d["rush_att"].mean()), 1),
            "Rush yds med (p10–p90)": q(d["rush_yds"]),
            "Targets": round(float(d["targets"].mean()), 1),
            "Rec med (p10–p90)": q(d["receptions"]),
            "Rec yds med (p10–p90)": q(d["rec_yds"]),
            "Pass yds med (p10–p90)": q(d["pass_yds"]) if "pass_yds" in d else "",
            "ATD %": round(100 * p_td, 1),
            "_pid": pid, "_touches": float(d["rush_att"].mean() + d["targets"].mean()),
        })
    df = pd.DataFrame(rows).sort_values("_touches", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

pbp, pos, snaps, sch, inj = load_all(SEASON)
# each team's usual home venue = its most frequent stadium_id this season (neutral sites are rare)
home_stadium = sch.groupby("home_team")["stadium_id"].agg(lambda x: x.value_counts().index[0]).to_dict()
open_weeks = sorted(sch.loc[sch["result"].isna() & (sch["game_type"] == "REG"), "week"].unique())
with st.sidebar:
    st.title("EV+ Gambit v3")
    st.caption("NFL props · DraftKings Predictions")
    week = st.selectbox("Slate week", open_weeks or [18], index=0)
    margin_c = st.slider("Safety margin (¢ of edge required after fee)", 0, 6, 2,
                         help="Buffer for model error. Not estimated yet — set from your CLV log once it has a few hundred entries.")
    n_sims = st.select_slider("Simulations", [5000, 10000, 20000], value=10000)
    use_cal = st.checkbox("Apply calibration (fit on 2025, validated on 2026)", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Season {SEASON} · data through the latest nflverse update")

stv = get_state(SEASON, int(week))
cal = C.load()

tab_slate, tab_price, tab_ledger, tab_notes = st.tabs(["Slate", "Price a contract", "Ledger", "Model notes"])

# ---------------------------------------------------------------------------
# Slate
# ---------------------------------------------------------------------------

with tab_slate:
    games = sch[(sch["week"] == week) & sch["result"].isna()].copy()
    if games.empty:
        st.info("No open games this week.")
        st.stop()
    games["label"] = games.apply(
        lambda g: f"{g['away_team']} @ {g['home_team']} — {g['gameday']} {g['gametime']} ET"
                  + (f" | {g['home_team']} {-g['spread_line']:+.1f}, total {g['total_line']:.1f}" if pd.notna(g["spread_line"]) else " | no line yet"),
        axis=1)
    label = st.selectbox("Game", games["label"].tolist())
    g = games[games["label"] == label].iloc[0]

    c1, c2, c3 = st.columns(3)
    home_book = c1.number_input(f"{g['home_team']} spread (book notation: −3 = {g['home_team']} favored by 3)",
                                value=-float(g["spread_line"]) if pd.notna(g["spread_line"]) else 0.0, step=0.5)
    spread = -home_book  # internal convention (nflverse): + = home favored
    total = c2.number_input("Total", value=float(g["total_line"]) if pd.notna(g["total_line"]) else 44.0, step=0.5)
    roof = g["roof"] if isinstance(g["roof"], str) else "unknown"
    home_sid = home_stadium.get(g["home_team"])
    wx = {}
    if roof in W.INDOOR:
        default_wind = 0.0
        c3.caption(f"Roof: {roof} → wind not applied")
    elif roof == "unknown" and g["stadium_id"] in W.RETRACTABLE:
        default_wind = 0.0
        c3.caption("Retractable roof, status not yet published → wind defaults to 0. Override if it will be open.")
    else:
        wx = get_weather(g["home_team"], g["stadium_id"], g["gameday"], g["gametime"], home_sid)
        default_wind = float(wx.get("wind_mph", 0.0)) if "error" not in wx else 0.0
        if "error" in wx:
            c3.caption(f"Roof: {roof}. Forecast unavailable ({wx['error'][:60]}) — enter wind manually.")
        else:
            c3.caption(f"Roof: {roof} · kickoff forecast: {wx['wind_mph']:.0f} mph (gusts {wx['gust_mph']:.0f}), "
                       f"{wx['temp_f']:.0f}°F, precip {wx['precip_in']:.2f} in")
    wind = c3.number_input("Wind at kickoff (mph)", value=round(default_wind, 1), step=1.0,
                           help="Only wind enters the model: its effect on pass rate and yards/attempt is estimated from data.")

    wk_inj = inj[(inj["week"] == week) & inj["report_status"].isin(["Out", "Doubtful"])]
    team_inputs = {}
    cols = st.columns(2)
    for col, team, home in [(cols[0], g["away_team"], False), (cols[1], g["home_team"], True)]:
        with col:
            st.subheader(team)
            roster = stv.roster.get(team, [])
            label_of = {p: f"{stv.players.loc[p, 'name']} ({stv.players.loc[p, 'position']})" for p in roster}
            pid_of = {v: k for k, v in label_of.items()}
            default_out = [label_of[p] for p in roster if p in set(wk_inj["gsis_id"])]
            out_labels = st.multiselect("Out / inactive", list(label_of.values()), default=default_out,
                                        key=f"out_{team}", help="Defaults to this week's Out/Doubtful report. Volume redistributes within the position group.")
            outs = [pid_of[x] for x in out_labels]
            P = stv.players
            returning = P[(P["team"] == team) & P["group"].isin(["QB", "RB", "WR", "TE"])
                          & ~P.index.isin(roster) & (P["last_t"] >= (SEASON - 1) * 20)]
            ret_label = {p: f"{P.loc[p, 'name']} ({P.loc[p, 'position']})" for p in returning.index}
            ret_pid = {v: k for k, v in ret_label.items()}
            add_labels = st.multiselect("Returning from injury (not in recent games)", list(ret_label.values()),
                                        key=f"add_{team}", help="Priced from his pre-injury share; volume is taken from his position group.")
            adds = [ret_pid[x] for x in add_labels]
            for p in adds:
                label_of[p] = ret_label[p]
                pid_of[ret_label[p]] = p
            qb_ids = sorted([p for p in roster + adds if stv.players.loc[p, "group"] == "QB" and p not in outs],
                            key=lambda p: -stv.players.loc[p, "n_db"])   # default = most recent-weighted dropbacks
            qb_labels = [label_of[p] for p in qb_ids]
            if qb_labels:
                qb = pid_of[st.selectbox("Starting QB", qb_labels, key=f"qb_{team}")]
                # the other QBs don't play: their designed-run share moves to the starter
                outs = outs + [p for p in qb_ids if p != qb]
            else:
                qb = None
                st.warning("No active QB on the recent roster — passing yards will not be priced.")
            imp = D.implied_points(spread, total, home)
            st.caption(f"Implied points: {imp:.1f}")
            team_inputs[team] = M.GameInputs(team, "", imp, spread if home else -spread,
                                             0.0 if roof in W.INDOOR else wind, tuple(outs), qb, tuple(adds))

    if st.button("Simulate game", type="primary"):
        st.session_state["sim"] = {t: M.simulate_team(stv, gi, n=n_sims, seed=11 + i)
                                   for i, (t, gi) in enumerate(team_inputs.items())}
        st.session_state["sim_label"] = label

    if "sim" in st.session_state and st.session_state.get("sim_label") == label:
        for team, sims in st.session_state["sim"].items():
            meta = sims.get("meta", {})
            if "error" in meta:
                st.error(meta["error"])
                continue
            st.markdown(f"**{team}** · plays ≈ {meta['plays_mu']:.0f} · pass share {meta['pass_share_mean']:.0%} · "
                        f"expected TDs {meta['td_lambda']:.2f}")
            df = summarize(stv, sims, cal, use_cal)
            st.dataframe(df.drop(columns=["_pid", "_touches"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Price a contract
# ---------------------------------------------------------------------------

with tab_price:
    if "sim" not in st.session_state:
        st.info("Simulate a game on the Slate tab first.")
    else:
        sims_all = st.session_state["sim"]
        options = {f"{stv.players.loc[pid, 'name']} ({t})": (t, pid)
                   for t, s in sims_all.items() for pid in s if pid != "meta"}
        pick = options[st.selectbox("Player", list(options))]
        market = st.selectbox("Market", list(MARKETS))
        key, cal_key = MARKETS[market]
        samples = sims_all[pick[0]][pick[1]].get(key)
        if samples is None:
            st.warning("No samples for that market (passing yards exist only for the starting QB).")
        else:
            if key == "tds":
                thr = 1
                st.caption("Contract: 1+ touchdown (rushing or receiving)")
            else:
                thr = st.number_input("Contract threshold (X+). For a line of 62.5 enter 63.",
                                      min_value=1, value=max(1, int(np.median(samples))), step=1)
            p_raw = PR.prob_at_least(samples, thr)
            p = float(C.apply(p_raw, cal_key, cal)) if use_cal else p_raw
            se = PR.mc_se(p_raw, len(samples))
            margin = margin_c / 100
            yes_max, no_max = PR.max_buy_price(p, margin), PR.max_buy_price(1 - p, margin)

            m1, m2, m3 = st.columns(3)
            m1.metric("Model P(YES)", f"{p:.1%}", help=f"Raw simulation {p_raw:.1%} ± {se:.1%} (MC s.e.)"
                      + (" before calibration" if use_cal else ""))
            m2.metric("Max buy YES", f"{yes_max * 100:.0f}¢" if yes_max else "none")
            m3.metric("Max buy NO", f"{no_max * 100:.0f}¢" if no_max else "none")

            c1, c2 = st.columns(2)
            yes_px = c1.number_input("DK YES price (¢)", 1, 99, 50)
            no_px = c2.number_input("DK NO price (¢)", 1, 99, 50)
            ev_yes = PR.ev_per_contract(p, yes_px / 100)
            ev_no = PR.ev_per_contract(1 - p, no_px / 100)
            st.write(f"EV per contract after fee — YES: **{ev_yes * 100:+.1f}¢**, NO: **{ev_no * 100:+.1f}¢**")
            if max(ev_yes, ev_no) >= margin:
                side = "YES" if ev_yes >= ev_no else "NO"
                st.success(f"{side} clears fee + {margin_c}¢ margin.")
            else:
                st.info("Neither side clears fee + margin. Pass.")

            if key in ("tds", "receptions"):
                vc = pd.Series(samples).value_counts(normalize=True).sort_index()
                st.bar_chart(vc.rename("share of sims"))
            else:
                cnt, edges = np.histogram(samples, bins=30, range=(np.percentile(samples, 0.5), np.percentile(samples, 99.5)))
                st.bar_chart(pd.Series(cnt / cnt.sum(), index=np.round((edges[:-1] + edges[1:]) / 2).astype(int),
                                       name="share of sims"))

            if st.button("Log decision"):
                st.session_state.setdefault("ledger", []).append(dict(
                    logged_at=dt.datetime.now().isoformat(timespec="seconds"), game=st.session_state.get("sim_label"),
                    player=stv.players.loc[pick[1], "name"], market=market, threshold=thr,
                    p_model=round(p, 4), yes_price=yes_px, no_price=no_px, ev_yes_c=round(ev_yes * 100, 2),
                    ev_no_c=round(ev_no * 100, 2), side="", entry_price="", closing_price="", result=""))
                st.toast("Logged")

# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

with tab_ledger:
    st.write("Fill in **side**, **entry_price** and **closing_price** (¢, the last YES/NO price before kickoff) "
             "and **result** (1/0). Closing-line value = closing − entry for the side you bought: it is the "
             "fastest honest signal of edge, far quicker than win rate.")
    st.caption("Streamlit Cloud storage resets on restart — download the CSV after each session and re-upload it next time.")
    up = st.file_uploader("Load ledger CSV", type="csv")
    if up is not None and not st.session_state.get("ledger_loaded"):
        st.session_state["ledger"] = pd.read_csv(up).to_dict("records")
        st.session_state["ledger_loaded"] = True
    led = pd.DataFrame(st.session_state.get("ledger", []))
    if led.empty:
        st.info("No entries yet.")
    else:
        led = st.data_editor(led, num_rows="dynamic", width="stretch")
        st.session_state["ledger"] = led.to_dict("records")
        e = pd.to_numeric(led.get("entry_price"), errors="coerce")
        c = pd.to_numeric(led.get("closing_price"), errors="coerce")
        clv = (c - e).dropna()
        if len(clv):
            st.metric("Mean CLV (¢ per contract)", f"{clv.mean():+.2f}", help=f"n = {len(clv)}; s.e. ≈ {clv.std(ddof=1) / np.sqrt(len(clv)):.2f}¢" if len(clv) > 1 else "n = 1")
        buf = io.StringIO()
        led.to_csv(buf, index=False)
        st.download_button("Download ledger CSV", buf.getvalue(), "gambit_ledger.csv", "text/csv")

# ---------------------------------------------------------------------------
# Model notes
# ---------------------------------------------------------------------------

with tab_notes:
    st.markdown("""
**How a price is made**
1. Market spread/total → each team's implied points.
2. Implied points + spread + team tendency (+ wind) → plays, pass share, expected TDs (all fitted).
3. Each player's carry/target share: empirical-Bayes posterior with a role prior (RB1, WR2…), recency-weighted,
   plus estimated game-to-game volatility. OUT players' volume moves to his position group.
4. Yards per play: bootstrap from the player's own plays mixed with the league pool; the mixing weight is estimated
   (per-carry RB efficiency is mostly noise, so RBs lean heavily on the league pool).
5. TDs: simulated volume × the player's expected-TD rate per carry/target (goal-line role lives here), scaled so the
   team total matches the market.
6. Per-market logistic calibration (fit on 2025, checked on 2026).
""")
    try:
        bt = pd.read_csv("results/backtest_summary.csv")
        st.markdown("**Walk-forward backtest (contested thresholds; Brier score, lower is better).** "
                    "`diff_vs_hit` < 0 means the model beats the “how often has he hit it” baseline; brackets are "
                    "95% cluster-bootstrap CIs over player-games. This measures probability quality, not profit.")
        st.dataframe(bt, hide_index=True, width="stretch")
    except FileNotFoundError:
        pass
    st.markdown("**Fitted at this cutoff**")
    st.json(stv.diagnostics, expanded=False)

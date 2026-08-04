"""
EV+ GAMBIT v2.1 — Streamlit Web App
Full prop scanner: K, Hits, TB, HR, RBI, Runs, SB, NRFI/YRFI, Totals
Hosted at: ev-gambit.streamlit.app
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import time

# Import the engine
from ev_gambit_v2 import (
    get_todays_games, get_pitcher_season_stats, get_stadium_weather,
    MLB_STADIUMS, implied_probability, calculate_ev, bayesian_hit_rate,
    evaluate_k_prop, sweep_alt_lines, parlay_odds, apply_boost,
    parlay_ev, rr_by_2s, get_lineup, scan_full_slate, rank_k_props
)

# Import extended prop evaluators
from prop_extensions import (
    get_batter_season_stats, get_pitcher_extended_stats,
    evaluate_hits_prop, evaluate_tb_prop, evaluate_hr_prop,
    evaluate_rbi_prop, evaluate_runs_prop, evaluate_sb_prop,
    evaluate_hits_allowed_prop, evaluate_walks_allowed_prop,
    evaluate_er_prop, evaluate_outs_prop, evaluate_nrfi,
    evaluate_game_total, fetch_all_props, scan_all_props,
    evaluate_single_player_prop, get_all_prop_types, search_player_id
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EV+ Gambit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .positive-ev {
        color: #00ff88;
        font-weight: 700;
    }
    .negative-ev {
        color: #ff4757;
        font-weight: 700;
    }
    .edge-high {
        background-color: #00ff8820;
        border-left: 4px solid #00ff88;
        padding: 8px 12px;
        border-radius: 4px;
    }
    .edge-low {
        background-color: #ff475720;
        border-left: 4px solid #ff4757;
        padding: 8px 12px;
        border-radius: 4px;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
    .stDataFrame {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/target.png", width=60)
    st.title("EV+ GAMBIT")
    st.caption("v2.1 — Full Prop Analytics Engine")
    st.divider()
    
    # API Key — pull from secrets first, fallback to text input
    api_key = st.secrets.get("ODDS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("🔑 Odds API Key", type="password", 
                                help="Get free key at the-odds-api.com")
    if api_key:
        st.success("🔑 Key loaded")
    else:
        st.warning("Add key in Settings → Secrets")
    
    st.divider()
    
    # Navigation
    page = st.radio("📍 Navigation", [
        "🎯 Daily K Scanner",
        "⚡ Full Prop Scanner",
        "🔍 Single Prop Analysis",
        "📊 Alt-Line Sweeper",
        "💰 Parlay Builder",
        "🔧 Boost Optimizer",
        "🌤️ Weather Check",
        "📋 Lineup Checker"
    ])
    
    st.divider()
    st.caption(f"📅 {datetime.date.today().strftime('%A, %B %d, %Y')}")
    st.caption("Built by Sean Rogers")
    st.caption("Go Cocks!!! #ForeverToThee")


# ============================================================
# ODDS API INTEGRATION
# ============================================================

def fetch_k_props(api_key, sport='baseball_mlb'):
    """Fetch live K props from The Odds API."""
    events_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events'
    try:
        events_res = requests.get(events_url, params={'apiKey': api_key})
        events_res.raise_for_status()
        events = events_res.json()
    except Exception as e:
        st.error(f"Events API error: {e}")
        return pd.DataFrame()
    
    odds_data = []
    progress = st.progress(0, text="Fetching live K props...")
    
    for i, event in enumerate(events[:15]):
        progress.progress((i + 1) / min(len(events), 15), 
                         text=f"Scanning {event.get('away_team', '')} @ {event.get('home_team', '')}...")
        event_id = event.get('id')
        if not event_id:
            continue
        
        odds_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds'
        odds_params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'pitcher_strikeouts,batter_hits,batter_home_runs,batter_total_bases',
            'oddsFormat': 'american',
            'bookmakers': 'draftkings,fanduel'
        }
        try:
            odds_res = requests.get(odds_url, params=odds_params)
            if odds_res.status_code == 200:
                for bookmaker in odds_res.json().get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'event': f"{event.get('away_team', '')} @ {event.get('home_team', '')}",
                                'player_name': outcome.get('description', outcome.get('name', 'Unknown')),
                                'prop_type': market.get('key'),
                                'prop_line': outcome.get('point', 0.0),
                                'odds': outcome.get('price'),
                                'direction': outcome.get('name', '').lower(),
                                'book': bookmaker.get('key', 'unknown'),
                                'game_time': event.get('commence_time', ''),
                            })
        except:
            continue
        time.sleep(0.5)
    
    progress.empty()
    return pd.DataFrame(odds_data)


# ============================================================
# PAGE: DAILY K SCANNER
# ============================================================

if "Daily K Scanner" in page:
    st.title("🎯 Daily K Scanner")
    st.caption("Enumerate the full population. Then evaluate.")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        scan_button = st.button("🔄 Scan Full Slate", type="primary", use_container_width=True)
    
    if scan_button:
        with st.spinner("Scanning MLB Stats API for all starters..."):
            slate = scan_full_slate(verbose=False)
        
        if not slate.empty:
            st.success(f"✅ {len(slate)} pitcher slots enumerated across {len(slate)//2} games")
            
            # Display slate
            st.subheader("📋 Full Slate")
            display_df = slate[slate['pitcher'] != 'TBD'][
                ['team', 'opponent', 'pitcher', 'era', 'whip', 'k_total', 
                 'k_per_start', 'k_per_9', 'games_started', 'venue', 'k_factor']
            ].copy()
            display_df.columns = ['Team', 'Opp', 'Pitcher', 'ERA', 'WHIP', 'K', 
                                  'K/GS', 'K/9', 'GS', 'Venue', 'K Factor']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Rankings
            st.subheader("📊 K Prop Rankings (estimated lines)")
            rankings = rank_k_props(slate)
            if not rankings.empty:
                rank_cols = [c for c in ['pitcher', 'team', 'opponent', 'k_avg', 'line', 
                            'hit_rate', 'implied_prob', 'edge', 'ev_per_dollar', 'confidence'] 
                            if c in rankings.columns]
                st.dataframe(rankings[rank_cols].head(20), use_container_width=True, hide_index=True)
            
            # Store in session
            st.session_state['slate'] = slate
            st.session_state['rankings'] = rankings
        else:
            st.warning("No games found for today.")
    
    # Live odds scanner
    if api_key:
        st.divider()
        st.subheader("🔴 Live Odds Scanner")
        if st.button("⚡ Scan Live K Props", use_container_width=True):
            odds_df = fetch_k_props(api_key)
            if not odds_df.empty:
                k_props = odds_df[odds_df['prop_type'] == 'pitcher_strikeouts'].copy()
                st.success(f"Found {len(k_props)} K prop lines")
                st.dataframe(k_props, use_container_width=True, hide_index=True)
                st.session_state['live_odds'] = k_props
            else:
                st.warning("No K props found. Check API key.")


# ============================================================
# PAGE: FULL PROP SCANNER (NEW)
# ============================================================

elif "Full Prop" in page:
    st.title("⚡ Full Prop Scanner")
    st.caption("Every prop. Every game. Every edge. One button.")
    
    if not api_key:
        st.error("Odds API key required. Add it in Settings → Secrets.")
    else:
        prop_filter = st.multiselect(
            "Filter prop types (leave empty for all)",
            ['pitcher_strikeouts', 'batter_hits', 'batter_total_bases', 
             'batter_home_runs', 'batter_rbis', 'batter_runs_scored',
             'batter_stolen_bases', 'pitcher_hits_allowed', 'pitcher_walks',
             'pitcher_earned_runs'],
            default=[]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_edge = st.number_input("Minimum edge %", value=0.0, step=1.0) / 100
        with col2:
            show_fades = st.checkbox("Show fades (negative EV)", value=False)
        
        if st.button("⚡ SCAN ALL PROPS", type="primary", use_container_width=True):
            with st.spinner("Pulling props from Odds API and evaluating against MLB stats..."):
                # Fetch all props
                props_df = fetch_all_props(api_key)
            
            if props_df.empty:
                st.warning("No props found. Games may not have started yet or API limit reached.")
            else:
                st.success(f"Found {len(props_df)} prop lines across {props_df['event'].nunique()} games")
                
                # Apply prop type filter
                if prop_filter:
                    props_df = props_df[props_df['prop_type'].isin(prop_filter)]
                
                # Evaluate each prop
                results = []
                games = get_todays_games()
                progress = st.progress(0, text="Evaluating props...")
                
                grouped = props_df.groupby(['player', 'prop_type', 'direction', 'line']).first().reset_index()
                total = len(grouped)
                
                for idx, row in grouped.iterrows():
                    progress.progress(min((idx + 1) / max(total, 1), 1.0),
                                     text=f"Evaluating {row['player']} {row['prop_type']}...")
                    try:
                        result = evaluate_single_player_prop(
                            row['player'], row['prop_type'],
                            row['line'], row['odds'], row['direction']
                        )
                        if result and 'error' not in result:
                            result['player'] = row.get('player', result.get('player', result.get('pitcher', 'Unknown')))
                            result['event'] = row['event']
                            result['prop_type'] = row['prop_type']
                            result['book'] = row['book']
                            result['odds'] = row['odds']
                            results.append(result)
                    except:
                        continue
                    time.sleep(0.15)
                
                progress.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('edge', ascending=False)
                    
                    # Add verdict
                    results_df['verdict'] = results_df['edge'].apply(
                        lambda x: '✅ PLAY' if x > 0.03 else '⚠️ MARGINAL' if x > 0 else '❌ FADE'
                    )
                    
                    # Filter by min edge
                    if not show_fades:
                        results_df = results_df[results_df['edge'] >= min_edge]
                    
                    display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                                   'line', 'odds', 'hit_rate', 'implied_prob', 'edge',
                                   'ev_per_dollar', 'confidence', 'verdict']
                                   if c in results_df.columns]
                    
                    st.subheader(f"📊 {len(results_df)} Props Evaluated")
                    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                    
                    # Top plays
                    plays = results_df[results_df['edge'] > 0.03]
                    if not plays.empty:
                        st.subheader("🏆 Top Plays (Edge > 3%)")
                        for _, play in plays.head(10).iterrows():
                            st.write(f"**{play.get('player', 'Unknown')}** — {play.get('direction', '').upper()} {play.get('line', '')} "
                                    f"({play.get('prop_type', '')}) | Edge: {play.get('edge', 0):+.1%} | "
                                    f"EV: ${play.get('ev_per_dollar', 0):+.3f} | {play.get('verdict', '')}")
                else:
                    st.warning("No props could be evaluated. Player lookups may have failed.")


# ============================================================
# PAGE: SINGLE PROP ANALYSIS
# ============================================================

elif "Single Prop" in page:
    st.title("🔍 Single Prop Analysis")
    st.caption("Evaluate any player prop — batter or pitcher")
    
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player Name", placeholder="Ty France, Tarik Skubal, etc.")
    with col2:
        prop_type = st.selectbox("Prop Type", get_all_prop_types(),
                                 format_func=lambda x: x.replace('_', ' ').title())
    
    col3, col4, col5 = st.columns(3)
    with col3:
        prop_line = st.number_input("Line", min_value=0.0, max_value=20.0, value=1.5, step=0.5)
    with col4:
        odds = st.number_input("American Odds", min_value=-500, max_value=500, value=-110)
    with col5:
        direction = st.selectbox("Direction", ["over", "under"])
    
    if st.button("📊 Analyze", type="primary"):
        if player_name:
            with st.spinner(f"Looking up {player_name}..."):
                result = evaluate_single_player_prop(player_name, prop_type, prop_line, odds, direction)
            
            if result and 'error' not in result:
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("True Probability", f"{result.get('hit_rate', 0):.1%}")
                with col2:
                    st.metric("Implied Probability", f"{result.get('implied_prob', 0):.1%}")
                with col3:
                    edge = result.get('edge', 0)
                    verdict = 'PLAY' if edge > 0.03 else 'FADE' if edge < 0 else 'MARGINAL'
                    st.metric("Edge", f"{edge:+.1%}", delta=verdict)
                with col4:
                    st.metric("EV per $1", f"${result.get('ev_per_dollar', 0):+.3f}")
                
                st.metric("Confidence", result.get('confidence', 'unknown').upper())
                
                # Show stat-specific info
                stat_avg = result.get('stat_avg', result.get('k_avg', result.get('hr_pg', result.get('sb_pg', 'N/A'))))
                st.metric("Player Average", f"{stat_avg}")
                st.metric("Separation from line", f"{result.get('separation', 'N/A')}")
                
                if edge > 0.03:
                    st.success(f"✅ Positive EV: {edge:+.1%} edge — PLAY")
                elif edge > 0:
                    st.warning(f"⚠️ Marginal: {edge:+.1%} edge")
                else:
                    st.error(f"❌ Negative EV: {edge:+.1%} edge — FADE")
            elif result and 'error' in result:
                st.error(result['error'])
            else:
                st.error("Could not evaluate prop. Check player name and prop type.")


# ============================================================
# PAGE: ALT-LINE SWEEPER
# ============================================================

elif "Alt-Line" in page:
    st.title("📊 Alt-Line Sweeper")
    st.caption("Find the optimal K line across all alt thresholds")
    
    pitcher_name = st.text_input("Pitcher Name", placeholder="Jacob Misiorowski")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        k_avg = st.number_input("K per Start", value=9.25, step=0.25)
    with col2:
        gs = st.number_input("Games Started", value=20, step=1)
    with col3:
        total_k = st.number_input("Total Ks", value=185, step=1)
    
    st.subheader("Enter Alt Lines (line : odds)")
    num_lines = st.number_input("Number of alt lines", min_value=2, max_value=10, value=5)
    
    odds_map = {}
    cols = st.columns(num_lines)
    for i in range(num_lines):
        with cols[i]:
            line = st.number_input(f"Line {i+1}", value=5.5 + i, step=0.5, key=f"line_{i}")
            line_odds = st.number_input(f"Odds {i+1}", value=-150 + (i * 60), step=5, key=f"odds_{i}")
            odds_map[line] = line_odds
    
    direction = st.selectbox("Direction", ["over", "under"], key="sweep_dir")
    park_k = st.number_input("Park K Factor", value=1.0, step=0.01)
    
    if st.button("🔍 Sweep All Lines", type="primary"):
        stats = {'name': pitcher_name, 'k_per_start': k_avg, 'games_started': gs, 'strikeouts': total_k}
        results = sweep_alt_lines(stats, odds_map, direction, park_k)
        
        if not results.empty:
            display_cols = [c for c in ['line', 'k_avg', 'hit_rate', 'implied_prob', 'edge', 
                           'ev_per_dollar', 'kelly_fraction', 'confidence'] if c in results.columns]
            st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
            
            best = results.iloc[0]
            if best['edge'] > 0:
                st.success(f"🏆 Best play: {direction.upper()} {best['line']} at EV ${best['ev_per_dollar']:+.3f}/dollar")
            else:
                st.warning("No positive EV lines found. Pass on this prop.")


# ============================================================
# PAGE: PARLAY BUILDER
# ============================================================

elif "Parlay" in page:
    st.title("💰 Parlay Builder")
    
    num_legs = st.number_input("Number of legs", min_value=2, max_value=10, value=4)
    
    legs_data = []
    for i in range(num_legs):
        st.subheader(f"Leg {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(f"Description", key=f"leg_name_{i}", placeholder="Miz U9.5")
        with col2:
            leg_odds = st.number_input(f"Odds", key=f"leg_odds_{i}", value=-110, step=5)
        with col3:
            true_prob = st.number_input(f"True Prob", key=f"leg_prob_{i}", value=0.55, 
                                        min_value=0.01, max_value=0.99, step=0.01)
        legs_data.append({'name': name, 'odds': leg_odds, 'true_prob': true_prob})
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        boost_pct = st.number_input("Boost %", min_value=0, max_value=200, value=0, step=5)
    with col2:
        stake = st.number_input("Stake ($)", min_value=1, max_value=1000, value=25, step=5)
    
    if st.button("📊 Calculate", type="primary"):
        result = parlay_ev(legs_data, boost_pct)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Parlay Odds", f"+{result['parlay_odds']}" if result['parlay_odds'] > 0 else str(result['parlay_odds']))
        with col2:
            if result['boosted_odds']:
                st.metric("Boosted Odds", f"+{result['boosted_odds']}" if result['boosted_odds'] > 0 else str(result['boosted_odds']))
        with col3:
            st.metric("True Probability", f"{result['combined_true_prob']:.1%}")
        with col4:
            st.metric("EV per $1", f"${result['ev_per_dollar']:+.3f}")
        
        payout = stake * (result['profit_if_win'] + 1)
        st.metric("💰 Potential Payout", f"${payout:.2f}")
        
        if result['edge'] > 0:
            st.success(f"✅ Positive EV: {result['edge']:+.1%} edge")
        else:
            st.error(f"❌ Negative EV: {result['edge']:+.1%} edge")
    
    # RR Calculator
    st.divider()
    st.subheader("🔄 Round Robin by 2s")
    
    if st.button("Calculate RR by 2s"):
        rr = rr_by_2s(legs_data, stake / len(legs_data))
        rr_cols = [c for c in ['combo', 'combined_true_prob', 'parlay_odds', 'edge', 
                   'ev_per_dollar', 'potential_profit'] if c in rr.columns]
        st.dataframe(rr[rr_cols], use_container_width=True, hide_index=True)
        
        total_ev = rr['ev_per_dollar'].sum() * (stake / len(legs_data))
        st.metric("Total RR EV", f"${total_ev:+.2f}")


# ============================================================
# PAGE: BOOST OPTIMIZER
# ============================================================

elif "Boost" in page:
    st.title("🔧 Boost Optimizer")
    st.caption("Find the optimal boost allocation across your available promotions")
    
    # Available boosts
    st.subheader("Available Boosts")
    num_boosts = st.number_input("Number of boosts", min_value=1, max_value=5, value=3)
    
    boosts = []
    for i in range(num_boosts):
        col1, col2, col3 = st.columns(3)
        with col1:
            b_name = st.text_input(f"Boost name", key=f"boost_name_{i}", placeholder="50% MLB Parlay")
        with col2:
            b_pct = st.number_input(f"Boost %", key=f"boost_pct_{i}", value=50, step=5)
        with col3:
            b_min = st.number_input(f"Min odds", key=f"boost_min_{i}", value=-200, step=10)
        boosts.append({'name': b_name or f'Boost {i+1}', 'pct': b_pct, 'min_odds': b_min})
    
    st.divider()
    
    # Legs
    st.subheader("Parlay Legs")
    num_legs = st.number_input("Legs", min_value=2, max_value=8, value=4, key="boost_legs")
    
    legs = []
    for i in range(num_legs):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(f"Leg", key=f"bl_name_{i}")
        with col2:
            odds = st.number_input(f"Odds", key=f"bl_odds_{i}", value=-110, step=5)
        with col3:
            prob = st.number_input(f"True Prob", key=f"bl_prob_{i}", value=0.55, step=0.01)
        legs.append({'name': name, 'odds': odds, 'true_prob': prob})
    
    if st.button("🏆 Find Optimal Boost", type="primary"):
        results = []
        for boost in boosts:
            ev = parlay_ev(legs, boost['pct'])
            ev['boost_name'] = boost['name']
            ev['boost_pct'] = boost['pct']
            results.append(ev)
        
        # No boost baseline
        ev_none = parlay_ev(legs, 0)
        ev_none['boost_name'] = 'No Boost'
        ev_none['boost_pct'] = 0
        results.append(ev_none)
        
        df = pd.DataFrame(results).sort_values('ev_per_dollar', ascending=False)
        
        display = [c for c in ['boost_name', 'boost_pct', 'parlay_odds', 'boosted_odds',
                   'combined_true_prob', 'edge', 'ev_per_dollar'] if c in df.columns]
        st.dataframe(df[display], use_container_width=True, hide_index=True)
        
        best = df.iloc[0]
        st.success(f"🏆 Best: {best['boost_name']} → EV: ${best['ev_per_dollar']:+.3f}/dollar")


# ============================================================
# PAGE: WEATHER CHECK
# ============================================================

elif "Weather" in page:
    st.title("🌤️ Stadium Weather")
    
    team = st.selectbox("Select Stadium (home team)", list(MLB_STADIUMS.keys()))
    
    if st.button("Check Weather", type="primary"):
        weather = get_stadium_weather(team)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperature", f"{weather.get('temp_f', 'N/A')}°F")
            st.metric("💨 Wind", f"{weather.get('wind_mph', 'N/A')} mph {weather.get('wind_dir', '')}")
        with col2:
            st.metric("💧 Humidity", f"{weather.get('humidity', 'N/A')}%")
            st.metric("🌧️ Precip", f"{weather.get('precip_mm', '0')} mm")
        with col3:
            st.metric("🏟️ Venue", weather.get('venue', 'Unknown'))
            st.metric("Conditions", weather.get('condition', 'Unknown'))
            if weather.get('indoor'):
                st.info("Indoor — Climate Controlled")


# ============================================================
# PAGE: LINEUP CHECKER
# ============================================================

elif "Lineup" in page:
    st.title("📋 Lineup Checker")
    st.caption("Verify who's actually batting before placing K props")
    
    if st.button("Load Today's Games", type="primary"):
        games = get_todays_games()
        if not games.empty:
            st.session_state['games'] = games
            game_options = [f"{g['away_team']} @ {g['home_team']} ({g['status']})" 
                          for _, g in games.iterrows()]
            st.session_state['game_options'] = game_options
    
    if 'games' in st.session_state:
        selected = st.selectbox("Select Game", st.session_state['game_options'])
        idx = st.session_state['game_options'].index(selected)
        game_id = st.session_state['games'].iloc[idx]['game_id']
        
        if st.button("📋 Load Lineup"):
            lineups = get_lineup(game_id)
            if lineups:
                for team, players in lineups.items():
                    st.subheader(f"{team} Lineup")
                    for i, p in enumerate(players, 1):
                        st.write(f"{i}. **{p['name']}** ({p['position']}) — {p['batting_avg']}")
            else:
                st.warning("Lineup not posted yet. Check closer to game time.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("EV+ Gambit v2.1 | The Cardinal Rule: Enumerate before you evaluate.")

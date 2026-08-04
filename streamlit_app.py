"""
EV+ GAMBIT v2.0 — Streamlit Web App
One-click deployment: streamlit run streamlit_app.py
Hosted deployment: push to GitHub → connect Streamlit Cloud

Place ev_gambit_v2.py in the same directory.
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
    st.caption("v2.0 — Prop Analytics Engine")
    st.divider()
    
    # API Key
    api_key = st.text_input("🔑 Odds API Key", type="password", 
                            help="Get free key at the-odds-api.com")
    if api_key:
        st.success("Key loaded")
    else:
        st.warning("Add key for live odds scanning")
    
    st.divider()
    
    # Navigation
    page = st.radio("📍 Navigation", [
        "🎯 Daily K Scanner",
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
# PAGE: SINGLE PROP ANALYSIS
# ============================================================

elif "Single Prop" in page:
    st.title("🔍 Single Prop Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pitcher_name = st.text_input("Pitcher Name", placeholder="Jacob Misiorowski")
    with col2:
        prop_line = st.number_input("K Line", min_value=0.5, max_value=20.0, value=6.5, step=0.5)
    with col3:
        odds = st.number_input("American Odds", min_value=-500, max_value=500, value=-110)
    with col4:
        direction = st.selectbox("Direction", ["over", "under"])
    
    col5, col6 = st.columns(2)
    with col5:
        park_team = st.selectbox("Ballpark (home team)", list(MLB_STADIUMS.keys()))
    with col6:
        k_factor = MLB_STADIUMS.get(park_team, {}).get('k_factor', 1.0)
        st.metric("Park K Factor", f"{k_factor:.2f}")
    
    if st.button("📊 Analyze", type="primary"):
        if pitcher_name:
            # Get pitcher stats from MLB API
            games = get_todays_games()
            pitcher_stats = None
            
            if not games.empty:
                for _, game in games.iterrows():
                    for side in ['away', 'home']:
                        if pitcher_name.lower() in game[f'{side}_pitcher_name'].lower():
                            pid = game[f'{side}_pitcher_id']
                            pitcher_stats = get_pitcher_season_stats(pid)
                            break
            
            if not pitcher_stats:
                st.warning("Pitcher not found in today's slate. Enter stats manually.")
                manual_col1, manual_col2, manual_col3 = st.columns(3)
                with manual_col1:
                    manual_k = st.number_input("K per Start", value=6.0, step=0.1)
                with manual_col2:
                    manual_gs = st.number_input("Games Started", value=15, step=1)
                with manual_col3:
                    manual_total_k = st.number_input("Total Ks", value=90, step=1)
                
                pitcher_stats = {
                    'name': pitcher_name,
                    'k_per_start': manual_k,
                    'games_started': manual_gs,
                    'strikeouts': manual_total_k,
                }
            
            if pitcher_stats:
                result = evaluate_k_prop(pitcher_stats, prop_line, odds, direction, k_factor)
                
                if 'error' not in result:
                    # Display results
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("True Probability", f"{result['hit_rate']:.1%}")
                    with col2:
                        st.metric("Implied Probability", f"{result['implied_prob']:.1%}")
                    with col3:
                        edge_color = "normal" if result['edge'] > 0 else "inverse"
                        st.metric("Edge", f"{result['edge']:+.1%}", 
                                 delta=f"{'PLAY' if result['edge'] > 0.03 else 'FADE' if result['edge'] < 0 else 'MARGINAL'}")
                    with col4:
                        st.metric("EV per $1", f"${result['ev_per_dollar']:+.3f}")
                    
                    st.metric("Confidence", result['confidence'].upper())
                    st.metric("K Average (park-adjusted)", f"{result['k_avg']:.1f}")
                    st.metric("Separation from line", f"{result['separation']:.1f}")
                else:
                    st.error(result['error'])


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
st.caption("EV+ Gambit v2.0 | The Cardinal Rule: Enumerate before you evaluate.")

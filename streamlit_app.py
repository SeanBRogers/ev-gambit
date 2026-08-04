"""
EV+ GAMBIT v2.2 — Streamlit Web App
Multi-sport prop scanner: MLB, NBA, WNBA, NFL
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

# Import multi-sport extensions
from sport_extensions import (
    fetch_sport_props, evaluate_any_sport_prop, get_sport_prop_types,
    scan_sport_props, SPORT_KEYS, SPORT_MARKETS
)

# Import arbitrage scanner
from arb_scanner import (
    scan_arbs, find_best_prices, calculate_arb_stakes, find_arb
)

# Import game-level odds
from game_analysis import (
    analyze_game, analyze_todays_games, get_team_record, get_team_id_from_abbr,
    calculate_win_probability, project_game_total
)

from game_odds import (
    fetch_game_odds, shop_game_lines, find_game_arbs, get_best_game_odds
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
    /* ============================================
       EV+ GAMBIT — Professional Dark Theme
       Inspired by Linemate.io
       ============================================ */
    
    /* Base */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    
    /* Headers */
    h1 { color: #f0f6fc; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #e6edf3; font-weight: 600; }
    h3 { color: #c9d1d9; font-weight: 600; }
    
    /* Play Cards */
    .play-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        transition: border-color 0.2s;
    }
    .play-card:hover {
        border-color: #388bfd;
    }
    .play-card-high {
        border-left: 4px solid #3fb950;
    }
    .play-card-moderate {
        border-left: 4px solid #d29922;
    }
    .play-card-fade {
        border-left: 4px solid #f85149;
    }
    
    /* Edge Badges */
    .badge-play {
        background: #23863620;
        color: #3fb950;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-fade {
        background: #f8514920;
        color: #f85149;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-marginal {
        background: #d2992220;
        color: #d29922;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-book {
        background: #388bfd20;
        color: #388bfd;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    /* Confidence Indicators */
    .conf-high { color: #3fb950; font-weight: 700; }
    .conf-mod { color: #d29922; font-weight: 700; }
    .conf-low { color: #8b949e; }
    
    /* Odds Display */
    .odds-plus { color: #3fb950; font-weight: 700; font-size: 1.1em; }
    .odds-minus { color: #f85149; font-weight: 700; font-size: 1.1em; }
    
    /* Stats Table */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
        font-size: 0.9em;
    }
    .stat-label { color: #8b949e; }
    .stat-value { color: #e6edf3; font-weight: 600; }
    .stat-pct { color: #3fb950; font-weight: 600; }
    
    /* Bet Slip */
    .slip-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .slip-player {
        color: #f0f6fc;
        font-weight: 600;
        font-size: 0.95em;
    }
    .slip-detail {
        color: #8b949e;
        font-size: 0.8em;
    }
    .slip-odds {
        color: #3fb950;
        font-weight: 700;
        font-size: 1.05em;
    }
    
    /* Parlay Summary */
    .parlay-box {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #388bfd40;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    
    /* Data Tables */
    .stDataFrame {
        background-color: #161b22;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #388bfd 0%, #1f6feb 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #58a6ff 0%, #388bfd 100%);
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4em;
    }
    
    /* Dividers */
    hr {
        border-color: #21262d;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
    
    /* Success/Warning/Error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    
    /* Remove Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/target.png", width=60)
    st.title("EV+ GAMBIT")
    st.caption("v2.2 — Multi-Sport Prop Engine")
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
        "⚡ Full Prop Scanner",
        "🏀 NBA/WNBA Scanner",
        "🏈 NFL/NCAA Scanner",
        "🏒 NHL Scanner",
        "⚽ Soccer Scanner",
        "💎 Arbitrage Scanner",
        "🏟️ Game Lines & Totals",
        "📊 Alt-Line Sweeper",
        "🌤️ Weather Check",
        "📋 Lineup Checker"
    ])
    
    st.divider()
    st.caption(f"📅 {datetime.date.today().strftime('%A, %B %d, %Y')}")
    st.caption("Built by Sean Rogers")
    st.caption("Go Cocks!!! #ForeverToThee")

# ============================================================
# BET SLIP FUNCTIONS (must be defined before page logic)
# ============================================================

if 'bet_slip' not in st.session_state:
    st.session_state.bet_slip = []

def add_to_slip(play):
    """Add a play to the bet slip."""
    for existing in st.session_state.bet_slip:
        if (existing.get('player') == play.get('player') and 
            existing.get('prop_type') == play.get('prop_type') and
            existing.get('line') == play.get('line')):
            return False
    st.session_state.bet_slip.append(play)
    return True

def remove_from_slip(index):
    """Remove a play from the bet slip by index."""
    if 0 <= index < len(st.session_state.bet_slip):
        st.session_state.bet_slip.pop(index)

def clear_slip():
    """Clear the entire bet slip."""
    st.session_state.bet_slip = []




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
        
        # Display previous results if they exist (survives rerun from slip add)
        if 'scan_results' in st.session_state and not st.session_state.get('rescan', False):
            results_df = st.session_state['scan_results']
            
            # Add verdict
            if 'verdict' not in results_df.columns:
                results_df['verdict'] = results_df['edge'].apply(
                    lambda x: '✅ PLAY' if x > 0.03 else '⚠️ MARGINAL' if x > 0 else '❌ FADE'
                )
            
            # Filter by min edge
            display_results = results_df[results_df['edge'] >= min_edge] if not show_fades else results_df
            
            display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                           'line', 'best_odds', 'best_book', 'hit_rate', 'implied_prob', 
                           'edge', 'ev_per_dollar', 'confidence', 'verdict']
                           if c in display_results.columns]
            
            st.subheader(f"📊 {len(display_results)} Props (cached results)")
            st.dataframe(display_results[display_cols], use_container_width=True, hide_index=True)
            
            # Top plays with add to slip
            plays = display_results[
                (display_results['edge'] > 0.03) & 
                (display_results['confidence'].isin(['high', 'moderate']))
            ]
            if not plays.empty:
                st.subheader(f"🏆 Top Plays — {len(plays)} found (Edge > 3% + High/Moderate Confidence)")
                for play_idx, play in plays.head(10).iterrows():
                    conf_icon = '🟢' if play.get('confidence') == 'high' else '🟡'
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        st.write(f"{conf_icon} **{play.get('player', 'Unknown')}** — "
                                f"{play.get('direction', '').upper()} {play.get('line', '')} "
                                f"({play.get('prop_type', '')}) | "
                                f"Edge: {play.get('edge', 0):+.1%} | "
                                f"EV: ${play.get('ev_per_dollar', 0):+.3f} | "
                                f"**Bet @ {play.get('best_book', '').upper()} ({play.get('best_odds', 0):+d})** | "
                                f"Confidence: {play.get('confidence', '').upper()}")
                        all_bks = play.get('all_books', '')
                        if all_bks:
                            st.caption(f"    All books: {all_bks}")
                    with col_btn:
                        if st.button("➕", key=f"cached_slip_{play_idx}"):
                            added = add_to_slip(play.to_dict())
                            if added:
                                st.rerun()
        
        if st.button("⚡ SCAN ALL PROPS", type="primary", use_container_width=True):
            st.session_state['rescan'] = True
            with st.spinner("Pulling props from Odds API and evaluating against MLB stats..."):
                # Fetch all props (includes all books)
                props_df = fetch_all_props(api_key)
            
            if props_df.empty:
                st.warning("No props found. Games may not have started yet or API limit reached.")
            else:
                st.success(f"Found {len(props_df)} prop lines across {props_df['event'].nunique()} games")
                
                # Apply prop type filter
                if prop_filter:
                    props_df = props_df[props_df['prop_type'].isin(prop_filter)]
                
                # NESTED ANALYSIS: For each prop, find the BEST odds across all books
                # Then evaluate using those best odds, and tag which book to use
                results = []
                games = get_todays_games()
                progress = st.progress(0, text="Evaluating props with best available odds...")
                
                # Find best odds per prop: sort by odds descending, keep first per group
                best_odds = props_df.sort_values('odds', ascending=False).drop_duplicates(
                    subset=['player', 'prop_type', 'direction', 'line'], keep='first'
                ).reset_index(drop=True)
                
                # Build lookup of all books for each prop
                all_books_lookup = {}
                for (player, prop_type, direction, line), group in props_df.groupby(['player', 'prop_type', 'direction', 'line']):
                    books_str = ', '.join([f"{r['book']}({r['odds']:+d})" for _, r in group.sort_values('odds', ascending=False).iterrows()])
                    all_books_lookup[(player, prop_type, direction, line)] = books_str
                
                total = len(best_odds)
                
                for idx, row in best_odds.iterrows():
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
                            result['best_book'] = row['book']
                            result['best_odds'] = row['odds']
                            
                            # Attach all books pricing
                            key = (row['player'], row['prop_type'], row['direction'], row['line'])
                            result['all_books'] = all_books_lookup.get(key, row['book'])
                            
                            results.append(result)
                    except:
                        continue
                    time.sleep(0.15)
                
                progress.empty()
                st.session_state['rescan'] = False
                
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('edge', ascending=False)
                    st.session_state['scan_results'] = results_df
                    
                    # Add verdict
                    results_df['verdict'] = results_df['edge'].apply(
                        lambda x: '✅ PLAY' if x > 0.03 else '⚠️ MARGINAL' if x > 0 else '❌ FADE'
                    )
                    
                    # Filter by min edge
                    if not show_fades:
                        results_df = results_df[results_df['edge'] >= min_edge]
                    
                    display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                                   'line', 'best_odds', 'best_book', 'hit_rate', 'implied_prob', 
                                   'edge', 'ev_per_dollar', 'confidence', 'verdict']
                                   if c in results_df.columns]
                    
                    st.subheader(f"📊 {len(results_df)} Props Evaluated (using best available odds)")
                    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                    
                    # Top plays — must have BOTH edge AND confidence
                    plays = results_df[
                        (results_df['edge'] > 0.03) & 
                        (results_df['confidence'].isin(['high', 'moderate']))
                    ]
                    if not plays.empty:
                        st.subheader(f"🏆 Top Plays — {len(plays)} found")
                        st.caption("Edge > 3% + High/Moderate Confidence")
                        for play_idx, play in plays.head(10).iterrows():
                            conf = play.get('confidence', 'low')
                            card_class = 'play-card-high' if conf == 'high' else 'play-card-moderate'
                            edge = play.get('edge', 0)
                            odds = play.get('best_odds', 0)
                            odds_class = 'odds-plus' if odds > 0 else 'odds-minus'
                            conf_class = 'conf-high' if conf == 'high' else 'conf-mod'
                            prop_clean = play.get('prop_type', '').replace('pitcher_', '').replace('batter_', '').replace('player_', '').replace('_', ' ').title()
                            all_bks = play.get('all_books', '')
                            
                            card_html = f"""
                            <div class="play-card {card_class}">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <span style="color:#f0f6fc; font-weight:700; font-size:1.1em;">{play.get('player', 'Unknown')}</span>
                                        <span style="color:#8b949e; margin-left:8px;">{play.get('direction', '').upper()} {play.get('line', '')} {prop_clean}</span>
                                    </div>
                                    <div>
                                        <span class="{odds_class}">{odds:+d}</span>
                                        <span class="badge-book" style="margin-left:8px;">{play.get('best_book', '').upper()}</span>
                                    </div>
                                </div>
                                <div style="display:flex; gap:16px; margin-top:8px;">
                                    <div class="stat-row" style="border:none; padding:0;">
                                        <span class="stat-label">Edge</span>
                                        <span class="stat-pct" style="margin-left:6px;">{edge:+.1%}</span>
                                    </div>
                                    <div class="stat-row" style="border:none; padding:0;">
                                        <span class="stat-label">EV/$1</span>
                                        <span class="stat-value" style="margin-left:6px;">${play.get('ev_per_dollar', 0):+.3f}</span>
                                    </div>
                                    <div class="stat-row" style="border:none; padding:0;">
                                        <span class="stat-label">Confidence</span>
                                        <span class="{conf_class}" style="margin-left:6px;">{conf.upper()}</span>
                                    </div>
                                </div>
                                <div style="margin-top:6px; color:#8b949e; font-size:0.8em;">
                                    {f'All books: {all_bks}' if all_bks else ''}
                                </div>
                            </div>
                            """
                            
                            col_info, col_btn = st.columns([6, 1])
                            with col_info:
                                st.markdown(card_html, unsafe_allow_html=True)
                            with col_btn:
                                st.write("")
                                if st.button("➕", key=f"add_slip_{play_idx}"):
                                    added = add_to_slip(play.to_dict())
                                    if added:
                                        st.rerun()
                    else:
                        st.info("No high-confidence plays found. Check the full table for marginal edges.")
                else:
                    st.warning("No props could be evaluated. Player lookups may have failed.")


# ============================================================
# PAGE: NBA/WNBA SCANNER
# ============================================================

elif "NBA/WNBA" in page:
    st.title("🏀 NBA / WNBA Prop Scanner")
    
    league_choice = st.radio("League", ["WNBA", "NBA"], horizontal=True)
    sport_key = league_choice.lower()
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        # Single player lookup
        st.subheader("🔍 Single Player Prop")
        col1, col2 = st.columns(2)
        with col1:
            bb_player = st.text_input("Player Name", placeholder="A'ja Wilson, Sabrina Ionescu...",
                                      key="bb_player")
        with col2:
            bb_prop = st.selectbox("Prop Type", get_sport_prop_types(sport_key),
                                   format_func=lambda x: x.replace('player_', '').replace('_', ' ').title(),
                                   key="bb_prop")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            bb_line = st.number_input("Line", value=20.5, step=0.5, key="bb_line")
        with col4:
            bb_odds = st.number_input("Odds", value=-110, step=5, key="bb_odds")
        with col5:
            bb_dir = st.selectbox("Direction", ["over", "under"], key="bb_dir")
        
        if st.button("📊 Analyze Player", type="primary"):
            with st.spinner(f"Looking up {bb_player}..."):
                result = evaluate_any_sport_prop(bb_player, sport_key, bb_prop, bb_line, bb_odds, bb_dir)
            
            if result and 'error' not in result:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("True Prob", f"{result.get('hit_rate', 0):.1%}")
                with col2:
                    st.metric("Implied Prob", f"{result.get('implied_prob', 0):.1%}")
                with col3:
                    edge = result.get('edge', 0)
                    st.metric("Edge", f"{edge:+.1%}",
                             delta='PLAY' if edge > 0.03 else 'FADE' if edge < 0 else 'MARGINAL')
                with col4:
                    st.metric("EV/$1", f"${result.get('ev_per_dollar', 0):+.3f}")
            elif result:
                st.error(result.get('error', 'Unknown error'))
        
        # Full scan
        st.divider()
        st.subheader(f"⚡ Full {league_choice} Scan")
        
        if st.button(f"⚡ SCAN ALL {league_choice} PROPS", type="primary", use_container_width=True):
            with st.spinner(f"Scanning {league_choice} props..."):
                results_df = scan_sport_props(api_key, sport_key)
            
            if not results_df.empty:
                st.success(f"Evaluated {len(results_df)} props")
                
                display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                               'line', 'odds', 'hit_rate', 'implied_prob', 'edge',
                               'ev_per_dollar', 'confidence', 'verdict']
                               if c in results_df.columns]
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                
                plays = results_df[
                    (results_df['edge'] > 0.03) & 
                    (results_df['confidence'].isin(['high', 'moderate']))
                ]
                if not plays.empty:
                    st.subheader(f"🏆 Top Plays — {len(plays)} found")
                    for _, play in plays.head(10).iterrows():
                        conf_icon = '🟢' if play.get('confidence') == 'high' else '🟡'
                        st.write(f"{conf_icon} **{play.get('player', '')}** — {play.get('direction', '').upper()} "
                                f"{play.get('line', '')} ({play.get('prop_type', '').replace('player_', '')}) | "
                                f"Edge: {play.get('edge', 0):+.1%} | Confidence: {play.get('confidence', '').upper()}")
                else:
                    st.info("No high-confidence plays found. Check the full table.")


# ============================================================
# PAGE: NFL SCANNER
# ============================================================

elif "NFL/NCAA" in page:
    st.title("🏈 NFL / NCAA Football Prop Scanner")
    
    league_choice = st.radio("League", ["NFL", "NCAAF"], horizontal=True)
    sport_key = league_choice.lower()
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        # Single player lookup
        st.subheader("🔍 Single Player Prop")
        col1, col2 = st.columns(2)
        with col1:
            nfl_player = st.text_input("Player Name", placeholder="Patrick Mahomes, Derrick Henry...",
                                       key="nfl_player")
        with col2:
            nfl_prop = st.selectbox("Prop Type", get_sport_prop_types('nfl'),
                                    format_func=lambda x: x.replace('player_', '').replace('_', ' ').title(),
                                    key="nfl_prop")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            nfl_line = st.number_input("Line", value=250.5, step=0.5, key="nfl_line")
        with col4:
            nfl_odds = st.number_input("Odds", value=-110, step=5, key="nfl_odds")
        with col5:
            nfl_dir = st.selectbox("Direction", ["over", "under"], key="nfl_dir")
        
        if st.button("📊 Analyze Player", type="primary", key="nfl_analyze"):
            with st.spinner(f"Looking up {nfl_player}..."):
                result = evaluate_any_sport_prop(nfl_player, sport_key, nfl_prop, nfl_line, nfl_odds, nfl_dir)
            
            if result and 'error' not in result:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("True Prob", f"{result.get('hit_rate', 0):.1%}")
                with col2:
                    st.metric("Implied Prob", f"{result.get('implied_prob', 0):.1%}")
                with col3:
                    edge = result.get('edge', 0)
                    st.metric("Edge", f"{edge:+.1%}",
                             delta='PLAY' if edge > 0.03 else 'FADE' if edge < 0 else 'MARGINAL')
                with col4:
                    st.metric("EV/$1", f"${result.get('ev_per_dollar', 0):+.3f}")
            elif result:
                st.error(result.get('error', 'Unknown error'))
        
        # Full scan
        st.divider()
        st.subheader(f"⚡ Full {league_choice} Scan")
        st.caption("Best during regular season (September - January)")
        
        if st.button(f"⚡ SCAN ALL {league_choice} PROPS", type="primary", use_container_width=True):
            with st.spinner(f"Scanning {league_choice} props..."):
                results_df = scan_sport_props(api_key, sport_key)
            
            if not results_df.empty:
                st.success(f"Evaluated {len(results_df)} props")
                
                display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                               'line', 'odds', 'hit_rate', 'implied_prob', 'edge',
                               'ev_per_dollar', 'confidence', 'verdict']
                               if c in results_df.columns]
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                
                plays = results_df[
                    (results_df['edge'] > 0.03) & 
                    (results_df['confidence'].isin(['high', 'moderate']))
                ]
                if not plays.empty:
                    st.subheader(f"🏆 Top Plays — {len(plays)} found")
                    for _, play in plays.head(10).iterrows():
                        conf_icon = '🟢' if play.get('confidence') == 'high' else '🟡'
                        st.write(f"{conf_icon} **{play.get('player', '')}** — {play.get('direction', '').upper()} "
                                f"{play.get('line', '')} ({play.get('prop_type', '').replace('player_', '')}) | "
                                f"Edge: {play.get('edge', 0):+.1%} | Confidence: {play.get('confidence', '').upper()}")
                else:
                    st.info("No high-confidence plays found.")
            else:
                st.warning(f"No {league_choice} props found. Check if games are scheduled today.")


# ============================================================
# PAGE: NHL SCANNER
# ============================================================

elif "NHL" in page:
    st.title("🏒 NHL Prop Scanner")
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        st.subheader("⚡ Full NHL Scan")
        
        if st.button("⚡ SCAN ALL NHL PROPS", type="primary", use_container_width=True):
            with st.spinner("Scanning NHL props..."):
                results_df = scan_sport_props(api_key, 'nhl')
            
            if not results_df.empty:
                st.success(f"Evaluated {len(results_df)} props")
                display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                               'line', 'odds', 'hit_rate', 'implied_prob', 'edge',
                               'ev_per_dollar', 'confidence', 'verdict']
                               if c in results_df.columns]
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.warning("No NHL props found. Check if season is active.")


# ============================================================
# PAGE: SOCCER SCANNER
# ============================================================

elif "Soccer" in page:
    st.title("⚽ Soccer Prop Scanner")
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        soccer_league = st.selectbox("League", 
            ['epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1', 'mls', 'champions_league'],
            format_func=lambda x: {'epl': 'Premier League', 'la_liga': 'La Liga', 
                                   'bundesliga': 'Bundesliga', 'serie_a': 'Serie A',
                                   'ligue_1': 'Ligue 1', 'mls': 'MLS',
                                   'champions_league': 'Champions League'}.get(x, x))
        
        if st.button(f"⚡ SCAN ALL PROPS", type="primary", use_container_width=True, key="soccer_scan"):
            with st.spinner(f"Scanning {soccer_league.upper()} props..."):
                results_df = scan_sport_props(api_key, soccer_league)
            
            if not results_df.empty:
                st.success(f"Evaluated {len(results_df)} props")
                display_cols = [c for c in ['player', 'event', 'prop_type', 'direction',
                               'line', 'odds', 'hit_rate', 'implied_prob', 'edge',
                               'ev_per_dollar', 'confidence', 'verdict']
                               if c in results_df.columns]
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.warning(f"No {soccer_league.upper()} props found. Check if games are scheduled today.")


# ============================================================
# PAGE: ARBITRAGE SCANNER
# ============================================================

elif "Arbitrage" in page:
    st.title("💎 Arbitrage Scanner")
    st.caption("Find guaranteed-profit opportunities across books")
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            arb_sport = st.selectbox("Sport", ['baseball_mlb', 'basketball_nba', 'basketball_wnba', 'americanfootball_nfl'],
                                     format_func=lambda x: x.split('_')[-1].upper(),
                                     key="arb_sport")
        with col2:
            min_margin = st.number_input("Min margin %", value=0.0, step=0.1, key="arb_margin")
        
        if st.button("💎 SCAN FOR ARBS", type="primary", use_container_width=True):
            with st.spinner("Scanning cross-book price gaps..."):
                results = scan_arbs(api_key, arb_sport, min_margin)
            
            if not results.empty:
                real_arbs = results[results.get('arb_exists', False) == True]
                near_arbs = results[results.get('near_arb', False) == True]
                
                if not real_arbs.empty:
                    st.success(f"🤑 {len(real_arbs)} ARBITRAGE OPPORTUNITIES FOUND!")
                    for _, arb in real_arbs.iterrows():
                        st.subheader(f"💎 {arb['player']} — {arb['prop_type']} {arb['line']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Guaranteed ROI", f"{arb['roi_pct']}%")
                            st.metric("Margin", f"{arb['margin']}%")
                        with col2:
                            st.metric(f"OVER @ {arb['over_book']}", f"{arb['over_odds']:+d}")
                            st.metric("Stake (per $100)", f"${arb['over_stake']}")
                        with col3:
                            st.metric(f"UNDER @ {arb['under_book']}", f"{arb['under_odds']:+d}")
                            st.metric("Stake (per $100)", f"${arb['under_stake']}")
                        st.caption(f"Guaranteed profit: ${arb['guaranteed_profit']} per $100 wagered")
                        st.divider()
                else:
                    st.info("No pure arbitrage found. Books are priced efficiently right now.")
                
                if not near_arbs.empty:
                    st.subheader(f"🔍 {len(near_arbs)} Near-Arb Opportunities (margin < 2%)")
                    st.caption("Not guaranteed profit, but the vig is minimal — strong value on one side")
                    display_cols = [c for c in ['player', 'prop_type', 'line', 'event',
                                   'over_odds', 'over_book', 'under_odds', 'under_book',
                                   'total_implied', 'margin'] if c in near_arbs.columns]
                    st.dataframe(near_arbs[display_cols], use_container_width=True, hide_index=True)
            else:
                st.warning("No props found to compare. Check API key and sport selection.")
        
        # Manual arb calculator
        st.divider()
        st.subheader("🧮 Manual Arb Calculator")
        st.caption("Enter odds from two books to check for arbitrage")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            manual_over = st.number_input("Over odds (Book A)", value=150, step=5, key="manual_over")
        with col2:
            manual_under = st.number_input("Under odds (Book B)", value=-120, step=5, key="manual_under")
        with col3:
            manual_stake = st.number_input("Total stake ($)", value=100, step=10, key="manual_stake")
        
        if st.button("🧮 Check Arb", key="check_arb"):
            result = calculate_arb_stakes(manual_over, manual_under, manual_stake)
            
            if 'error' not in result:
                st.success(f"✅ ARBITRAGE EXISTS! Guaranteed profit: ${result['guaranteed_profit']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Bet OVER at {result['over_odds']:+d}", f"${result['over_bet']}")
                    st.metric("If over hits", f"${result['if_over_hits']}")
                with col2:
                    st.metric(f"Bet UNDER at {result['under_odds']:+d}", f"${result['under_bet']}")
                    st.metric("If under hits", f"${result['if_under_hits']}")
                st.metric("ROI", f"{result['roi_pct']}%")
            else:
                st.error(f"No arb. Combined implied: {result['total_implied']}%. Margin: {result['margin']}%")


# ============================================================
# PAGE: GAME LINES & TOTALS
# ============================================================

elif "Game Lines" in page:
    st.title("🏟️ Game Lines & Totals")
    st.caption("Matchup analysis, best lines, and arb detection — all in one view")
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        game_sport = st.selectbox("Sport", 
            ['mlb', 'nba', 'wnba', 'nfl', 'ncaaf', 'ncaab', 'nhl', 'epl', 'la_liga', 'bundesliga', 'serie_a', 'mls', 'champions_league'],
            format_func=lambda x: x.upper() if len(x) <= 4 else {
                'ncaaf': 'NCAA Football', 'ncaab': 'NCAA Basketball',
                'epl': 'Premier League', 'la_liga': 'La Liga',
                'bundesliga': 'Bundesliga', 'serie_a': 'Serie A',
                'champions_league': 'Champions League'
            }.get(x, x.upper()),
            key="game_sport")
        
        if st.button("🏟️ ANALYZE GAMES", type="primary", use_container_width=True):
            
            if True:  # Single unified view
                with st.spinner("Analyzing today's matchups..."):
                    if game_sport == 'mlb':
                        analyses = analyze_todays_games()
                    else:
                        analyses = []
                    
                    # Also fetch best odds for each game
                    game_odds_df = get_best_game_odds(api_key, game_sport)
                    
                    # And check for arbs
                    game_arbs_df = find_game_arbs(api_key, game_sport)
                    real_arbs = game_arbs_df[game_arbs_df.get('arb_exists', False) == True] if not game_arbs_df.empty else pd.DataFrame()
                
                if analyses:
                    st.success(f"Analyzed {len(analyses)} games")
                    
                    for analysis in analyses:
                        wp = analysis.get('win_probability', {})
                        tp = analysis.get('projected_total', {})
                        home = analysis.get('home_team', '?')
                        away = analysis.get('away_team', '?')
                        hr = analysis.get('home_record', {})
                        ar = analysis.get('away_record', {})
                        home_prob = wp.get('home_prob', 0.5)
                        away_prob = wp.get('away_prob', 0.5)
                        
                        # Game header
                        fav = home if home_prob > 0.55 else away if away_prob > 0.55 else 'Toss-up'
                        st.subheader(f"{away} @ {home}")
                        st.caption(f"{analysis.get('venue', '')} | Favored: {fav}")
                        
                        # Win probability row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            delta_a = f"L10: {ar.get('last10_wins', '?')}-{ar.get('last10_losses', '?')} {ar.get('streak', '')}"
                            st.metric(f"{away}", f"{away_prob:.0%}", delta=delta_a)
                            st.caption(f"Record: {ar.get('wins', 0)}-{ar.get('losses', 0)}")
                        with col2:
                            st.metric("Projected Total", f"{tp.get('projected_total', 'N/A')}")
                            st.caption(f"{away} {tp.get('away_runs_proj', '?')} — {home} {tp.get('home_runs_proj', '?')}")
                        with col3:
                            delta_h = f"L10: {hr.get('last10_wins', '?')}-{hr.get('last10_losses', '?')} {hr.get('streak', '')}"
                            st.metric(f"{home}", f"{home_prob:.0%}", delta=delta_h)
                            st.caption(f"Record: {hr.get('wins', 0)}-{hr.get('losses', 0)}")
                        
                        # Pitching matchup
                        st.write(f"**Pitching:** {analysis.get('away_pitcher', 'TBD')} ({analysis.get('away_era', 'N/A')} ERA) vs {analysis.get('home_pitcher', 'TBD')} ({analysis.get('home_era', 'N/A')} ERA)")
                        
                        # Reasoning
                        for reason in wp.get('reasons', []):
                            st.caption(f"→ {reason}")
                        
                        # Best available odds for this game
                        if not game_odds_df.empty:
                            game_str = f"{away} @ {home}" if away and home else ''
                            game_match = game_odds_df[
                                game_odds_df['game'].str.contains(home, case=False, na=False) |
                                game_odds_df['game'].str.contains(hr.get('team_name', 'ZZZZZ'), case=False, na=False)
                            ]
                            if not game_match.empty:
                                gm = game_match.iloc[0]
                                game_label = gm.get('game', f'{away} @ {home}')
                                st.write("**Best Available Lines:**")
                                
                                away_name = gm.get('away_team', away)
                                home_name = gm.get('home_team', home)
                                
                                # Header row
                                hdr = st.columns([1.5, 0.4, 1.5, 0.4, 1.5, 0.4])
                                with hdr[0]:
                                    st.caption("SPREAD")
                                with hdr[2]:
                                    st.caption("TOTAL")
                                with hdr[4]:
                                    st.caption("MONEYLINE")
                                
                                # Away row
                                ar1, ab1, ar2, ab2, ar3, ab3 = st.columns([1.5, 0.4, 1.5, 0.4, 1.5, 0.4])
                                with ar1:
                                    try:
                                        asp = float(gm.get('away_spread', 0))
                                        asp_odds = int(float(gm.get('away_spread_odds', 0)))
                                        st.write(f"**{away_name}** {asp:+.1f} ({asp_odds:+d})")
                                        st.caption(f"{gm.get('away_spread_book', '')}")
                                    except:
                                        st.write(f"**{away_name}** —")
                                with ab1:
                                    try:
                                        asp_o = int(float(gm.get('away_spread_odds', 0)))
                                        if st.button("➕", key=f"slip_asp_{home}_{away}"):
                                            add_to_slip({'player': away_name, 'prop_type': 'spread',
                                                'direction': f"{float(gm.get('away_spread', 0)):+.1f}",
                                                'line': gm.get('away_spread', ''), 'best_odds': asp_o,
                                                'best_book': gm.get('away_spread_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                with ar2:
                                    try:
                                        tot = gm.get('total_line', '')
                                        ov = int(float(gm.get('over_odds', 0)))
                                        st.write(f"O {tot} ({ov:+d})")
                                        st.caption(f"{gm.get('over_book', '')}")
                                    except:
                                        st.write("—")
                                with ab2:
                                    try:
                                        ov_o = int(float(gm.get('over_odds', 0)))
                                        if st.button("➕", key=f"slip_ov_{home}_{away}"):
                                            add_to_slip({'player': game_label, 'prop_type': 'game_total',
                                                'direction': 'over', 'line': gm.get('total_line', ''),
                                                'best_odds': ov_o, 'best_book': gm.get('over_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                with ar3:
                                    try:
                                        aml = int(float(gm.get('away_ml', 0)))
                                        st.write(f"**{aml:+d}**")
                                        st.caption(f"{gm.get('away_ml_book', '')}")
                                    except:
                                        st.write("—")
                                with ab3:
                                    try:
                                        aml_o = int(float(gm.get('away_ml', 0)))
                                        if st.button("➕", key=f"slip_aml_{home}_{away}"):
                                            add_to_slip({'player': away_name, 'prop_type': 'moneyline',
                                                'direction': 'ml', 'line': '',
                                                'best_odds': aml_o, 'best_book': gm.get('away_ml_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                
                                # Home row
                                hr1, hb1, hr2, hb2, hr3, hb3 = st.columns([1.5, 0.4, 1.5, 0.4, 1.5, 0.4])
                                with hr1:
                                    try:
                                        hsp = float(gm.get('home_spread', 0))
                                        hsp_odds = int(float(gm.get('home_spread_odds', 0)))
                                        st.write(f"**{home_name}** {hsp:+.1f} ({hsp_odds:+d})")
                                        st.caption(f"{gm.get('home_spread_book', '')}")
                                    except:
                                        st.write(f"**{home_name}** —")
                                with hb1:
                                    try:
                                        hsp_o = int(float(gm.get('home_spread_odds', 0)))
                                        if st.button("➕", key=f"slip_hsp_{home}_{away}"):
                                            add_to_slip({'player': home_name, 'prop_type': 'spread',
                                                'direction': f"{float(gm.get('home_spread', 0)):+.1f}",
                                                'line': gm.get('home_spread', ''), 'best_odds': hsp_o,
                                                'best_book': gm.get('home_spread_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                with hr2:
                                    try:
                                        tot = gm.get('total_line', '')
                                        un = int(float(gm.get('under_odds', 0)))
                                        st.write(f"U {tot} ({un:+d})")
                                        st.caption(f"{gm.get('under_book', '')}")
                                    except:
                                        st.write("—")
                                with hb2:
                                    try:
                                        un_o = int(float(gm.get('under_odds', 0)))
                                        if st.button("➕", key=f"slip_un_{home}_{away}"):
                                            add_to_slip({'player': game_label, 'prop_type': 'game_total',
                                                'direction': 'under', 'line': gm.get('total_line', ''),
                                                'best_odds': un_o, 'best_book': gm.get('under_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                with hr3:
                                    try:
                                        hml = int(float(gm.get('home_ml', 0)))
                                        st.write(f"**{hml:+d}**")
                                        st.caption(f"{gm.get('home_ml_book', '')}")
                                    except:
                                        st.write("—")
                                with hb3:
                                    try:
                                        hml_o = int(float(gm.get('home_ml', 0)))
                                        if st.button("➕", key=f"slip_hml_{home}_{away}"):
                                            add_to_slip({'player': home_name, 'prop_type': 'moneyline',
                                                'direction': 'ml', 'line': '',
                                                'best_odds': hml_o, 'best_book': gm.get('home_ml_book', ''),
                                                'edge': 0, 'event': game_label})
                                            st.rerun()
                                    except:
                                        pass
                                
                                # All books
                                away_all = gm.get('away_ml_all', '')
                                home_all = gm.get('home_ml_all', '')
                                if away_all:
                                    st.caption(f"{away_name}: {away_all}")
                                if home_all:
                                    st.caption(f"{home_name}: {home_all}")
                        
                        # Arb flag if exists for this game
                        if not real_arbs.empty:
                            game_arb = real_arbs[real_arbs['game'].str.contains(home, case=False, na=False)]
                            if not game_arb.empty:
                                for _, arb in game_arb.iterrows():
                                    roi = arb.get('roi_pct', 0)
                                    if roi > 5:
                                        st.warning(f"⚠️ Possible arb ({arb.get('market', '')}): {roi:.1f}% ROI — VERIFY on apps before placing")
                                    else:
                                        st.success(f"💎 Arb found ({arb.get('market', '')}): {roi:.1f}% ROI | {arb.get('side_1', '')} + {arb.get('side_2', '')}")
                        
                        st.divider()
                
                # Handle non-MLB sports (no analysis engine yet, just show odds)
                if not analyses and not game_odds_df.empty:
                    st.success(f"Found {len(game_odds_df)} games")
                    for _, gm in game_odds_df.iterrows():
                        st.subheader(gm.get('game', ''))
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            away_ml = gm.get('away_ml', 'N/A')
                            if isinstance(away_ml, (int, float)):
                                st.metric(f"{gm.get('away_team', '?')} ML", f"{away_ml:+.0f}", delta=gm.get('away_ml_book', ''))
                        with col2:
                            total = gm.get('total_line', '')
                            if total:
                                over_odds = gm.get('over_odds', '')
                                under_odds = gm.get('under_odds', '')
                                if isinstance(over_odds, (int, float)):
                                    st.metric(f"O/U {total}", f"O {over_odds:+.0f}")
                        with col3:
                            home_ml = gm.get('home_ml', 'N/A')
                            if isinstance(home_ml, (int, float)):
                                st.metric(f"{gm.get('home_team', '?')} ML", f"{home_ml:+.0f}", delta=gm.get('home_ml_book', ''))
                        st.divider()
            


# ============================================================
# PAGE: LINE SHOPPING
# ============================================================

elif "Line Shopping" in page:
    st.title("🛒 Line Shopping")
    st.caption("Find which book has the best price on every prop")
    
    if not api_key:
        st.error("Odds API key required.")
    else:
        shop_sport = st.selectbox("Sport", ['baseball_mlb', 'basketball_nba', 'basketball_wnba', 'americanfootball_nfl'],
                                  format_func=lambda x: x.split('_')[-1].upper(),
                                  key="shop_sport")
        
        if st.button("🛒 SHOP ALL LINES", type="primary", use_container_width=True):
            with st.spinner("Comparing prices across books..."):
                results = find_best_prices(api_key, shop_sport)
            
            if not results.empty:
                st.success(f"Compared {len(results)} props across {results['books_compared'].max()} books")
                
                # Biggest spreads first — most value in shopping
                st.subheader("📊 Biggest Price Gaps (shop here first)")
                display_cols = [c for c in ['player', 'prop_type', 'line', 'event',
                               'best_over_odds', 'best_over_book', 'worst_over_odds',
                               'over_spread', 'best_under_odds', 'best_under_book',
                               'worst_under_odds', 'under_spread'] if c in results.columns]
                st.dataframe(results[display_cols].head(30), use_container_width=True, hide_index=True)
                
                big_gaps = results[results['max_spread'] >= 15]
                if not big_gaps.empty:
                    st.subheader(f"🏆 {len(big_gaps)} props with 15+ point spread between books")
                    for _, row in big_gaps.head(10).iterrows():
                        st.write(f"**{row['player']}** — {row['prop_type']} {row['line']} | "
                                f"Best over: {row['best_over_odds']:+.0f} ({row['best_over_book']}) vs "
                                f"Worst: {row['worst_over_odds']:+.0f} ({row['worst_over_book']}) | "
                                f"Spread: {row['over_spread']:.0f} pts")
            else:
                st.warning("No props found to compare.")


# ============================================================
# PAGE: SINGLE PROP ANALYSIS (multi-sport)
# ============================================================

elif "Single Prop" in page:
    st.title("🔍 Single Prop Analysis")
    st.caption("Evaluate any prop — MLB, NBA, WNBA, or NFL")
    
    sport_select = st.radio("Sport", ["MLB", "NBA", "WNBA", "NFL"], horizontal=True)
    sport_key = sport_select.lower()
    
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player Name", placeholder="Ty France, A'ja Wilson, Patrick Mahomes...")
    with col2:
        prop_type = st.selectbox("Prop Type", get_sport_prop_types(sport_key),
                                 format_func=lambda x: x.replace('player_', '').replace('pitcher_', '').replace('batter_', '').replace('_', ' ').title())
    
    col3, col4, col5 = st.columns(3)
    with col3:
        default_line = 6.5 if sport_key == 'mlb' else 20.5 if sport_key in ['nba', 'wnba'] else 250.5
        prop_line = st.number_input("Line", min_value=0.0, max_value=500.0, value=default_line, step=0.5)
    with col4:
        odds = st.number_input("American Odds", min_value=-500, max_value=500, value=-110)
    with col5:
        direction = st.selectbox("Direction", ["over", "under"])
    
    if st.button("📊 Analyze", type="primary"):
        if player_name:
            with st.spinner(f"Looking up {player_name} ({sport_select})..."):
                result = evaluate_any_sport_prop(player_name, sport_key, prop_type, prop_line, odds, direction)
            
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
st.caption("EV+ Gambit v2.2 | MLB • NBA • WNBA • NFL | Enumerate before you evaluate.")

# Display bet slip in sidebar
with st.sidebar:
    st.divider()
    st.subheader(f"🎫 Bet Slip ({len(st.session_state.bet_slip)})")
    
    if st.session_state.bet_slip:
        total_implied = 1.0
        slip_text_lines = []
        
        for i, play in enumerate(st.session_state.bet_slip):
            player = play.get('player', 'Unknown')
            direction = play.get('direction', '').upper()
            line = play.get('line', '')
            odds = play.get('best_odds', play.get('odds', 0))
            book = play.get('best_book', play.get('book', ''))
            prop_type = play.get('prop_type', '').replace('pitcher_', '').replace('batter_', '').replace('player_', '').replace('_', ' ')
            edge = play.get('edge', 0)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{player}** {direction} {line} ({prop_type})")
                st.caption(f"{odds:+d} @ {book} | Edge: {edge:+.1%}")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    remove_from_slip(i)
                    st.rerun()
            
            # Track for parlay calc
            imp = implied_probability(odds) if odds != 0 else 0.5
            total_implied *= imp
            slip_text_lines.append(f"{player} {direction} {line} ({prop_type}) | {odds:+d} @ {book}")
        
        # Parlay math
        if len(st.session_state.bet_slip) >= 2:
            from ev_gambit_v2 import parlay_odds as calc_parlay
            odds_list = [p.get('best_odds', p.get('odds', -110)) for p in st.session_state.bet_slip]
            parlay_info = calc_parlay(odds_list)
            
            st.divider()
            st.metric("Parlay Odds", f"+{parlay_info['combined_american']}" if parlay_info['combined_american'] > 0 else str(parlay_info['combined_american']))
            
            stake = st.number_input("Stake $", value=25, step=5, key="slip_stake")
            if parlay_info['combined_american'] > 0:
                payout = stake * (1 + parlay_info['combined_american'] / 100)
            else:
                payout = stake * (1 + 100 / -parlay_info['combined_american'])
            st.metric("Payout", f"${payout:.2f}")
            
            # Boost
            slip_boost = st.number_input("Boost %", value=0, step=5, key="slip_boost")
            if slip_boost > 0:
                from ev_gambit_v2 import apply_boost
                boosted = apply_boost(parlay_info['combined_american'], slip_boost)
                st.metric("Boosted Odds", f"+{boosted['boosted_odds']}" if boosted['boosted_odds'] > 0 else str(boosted['boosted_odds']))
                if boosted['boosted_odds'] > 0:
                    boosted_payout = stake * (1 + boosted['boosted_odds'] / 100)
                else:
                    boosted_payout = stake * (1 + 100 / -boosted['boosted_odds'])
                st.metric("Boosted Payout", f"${boosted_payout:.2f}")
        
        # Copy slip
        slip_text = "\n".join(slip_text_lines)
        st.text_area("📋 Copy Slip", slip_text, height=100, key="copy_slip")
        
        # Group by book
        books_used = set()
        for p in st.session_state.bet_slip:
            books_used.add(p.get('best_book', p.get('book', 'unknown')))
        if len(books_used) > 1:
            st.warning(f"⚠️ Legs span {len(books_used)} books: {', '.join(books_used)}")
        
        if st.button("🗑️ Clear Slip", use_container_width=True):
            clear_slip()
            st.rerun()
    else:
        st.caption("Add plays from any scanner page")

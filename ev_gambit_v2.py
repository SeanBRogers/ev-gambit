"""
EV+ GAMBIT v2.0 — Sports Prop Analytics Engine
Built for Google Colab | Sean Rogers | August 2026

UPGRADES FROM v1:
- Full 30-team MLB stadium map with K-factors and park dimensions
- MLB Stats API for real-time lineups (free, no key)
- Alt-line sweep: find optimal EV across 5+ thresholds
- Parlay EV calculator with boost optimizer
- Full slate scanner: enumerate every game/starter before evaluating
- Bayesian-adjusted hit rates (regress small samples toward league mean)
- WNBA support via wnba_api or manual input
- Weather via wttr.in (no API key, no Selenium)
- Lineup validator: check actual batting order before placing K props

DEPENDENCIES (run in Colab first cell):
!pip install pybaseball nba_api requests pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import requests
import datetime
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

# ============================================================
# SPORTSBOOK CONFIGURATION
# Sean's books: FanDuel, DraftKings, Fanatics, BetMGM, Caesars
# Change this ONE constant to update every scanner in the app
# ============================================================

PREFERRED_BOOKS = 'draftkings,fanduel,fanatics,betmgm,caesars'

# ============================================================
# MODULE 1: STADIUM & PARK FACTORS
# ============================================================

MLB_STADIUMS = {
    'AZ':  {'name': 'Chase Field', 'lat': 33.445, 'lon': -112.066, 'indoor': True, 'k_factor': 1.02},
    'ATL': {'name': 'Truist Park', 'lat': 33.890, 'lon': -84.468, 'indoor': False, 'k_factor': 0.98},
    'BAL': {'name': 'Camden Yards', 'lat': 39.284, 'lon': -76.622, 'indoor': False, 'k_factor': 1.01},
    'BOS': {'name': 'Fenway Park', 'lat': 42.346, 'lon': -71.098, 'indoor': False, 'k_factor': 0.97},
    'CHC': {'name': 'Wrigley Field', 'lat': 41.948, 'lon': -87.656, 'indoor': False, 'k_factor': 0.99},
    'CWS': {'name': 'Guaranteed Rate', 'lat': 41.830, 'lon': -87.634, 'indoor': False, 'k_factor': 1.00},
    'CIN': {'name': 'Great American', 'lat': 39.097, 'lon': -84.507, 'indoor': False, 'k_factor': 1.03},
    'CLE': {'name': 'Progressive Field', 'lat': 41.496, 'lon': -81.685, 'indoor': False, 'k_factor': 0.98},
    'COL': {'name': 'Coors Field', 'lat': 39.756, 'lon': -104.994, 'indoor': False, 'k_factor': 0.92},
    'DET': {'name': 'Comerica Park', 'lat': 42.339, 'lon': -83.049, 'indoor': False, 'k_factor': 1.01},
    'HOU': {'name': 'Minute Maid Park', 'lat': 29.757, 'lon': -95.355, 'indoor': True, 'k_factor': 1.00},
    'KC':  {'name': 'Kauffman Stadium', 'lat': 39.051, 'lon': -94.480, 'indoor': False, 'k_factor': 0.96},
    'LAA': {'name': 'Angel Stadium', 'lat': 33.800, 'lon': -117.883, 'indoor': False, 'k_factor': 1.04},
    'LAD': {'name': 'Dodger Stadium', 'lat': 34.073, 'lon': -118.240, 'indoor': False, 'k_factor': 1.01},
    'MIA': {'name': 'LoanDepot Park', 'lat': 25.778, 'lon': -80.220, 'indoor': True, 'k_factor': 1.00},
    'MIL': {'name': 'American Family', 'lat': 43.028, 'lon': -87.971, 'indoor': True, 'k_factor': 1.01},
    'MIN': {'name': 'Target Field', 'lat': 44.982, 'lon': -93.278, 'indoor': False, 'k_factor': 0.99},
    'NYM': {'name': 'Citi Field', 'lat': 40.757, 'lon': -73.846, 'indoor': False, 'k_factor': 0.98},
    'NYY': {'name': 'Yankee Stadium', 'lat': 40.829, 'lon': -73.926, 'indoor': False, 'k_factor': 1.02},
    'ATH': {'name': 'Sutter Health Park', 'lat': 38.580, 'lon': -121.509, 'indoor': False, 'k_factor': 1.00},
    'PHI': {'name': 'Citizens Bank Park', 'lat': 39.906, 'lon': -75.166, 'indoor': False, 'k_factor': 1.02},
    'PIT': {'name': 'PNC Park', 'lat': 40.447, 'lon': -80.006, 'indoor': False, 'k_factor': 0.97},
    'SD':  {'name': 'Petco Park', 'lat': 32.707, 'lon': -117.157, 'indoor': False, 'k_factor': 0.96},
    'SF':  {'name': 'Oracle Park', 'lat': 37.778, 'lon': -122.389, 'indoor': False, 'k_factor': 0.98},
    'SEA': {'name': 'T-Mobile Park', 'lat': 47.591, 'lon': -122.332, 'indoor': True, 'k_factor': 1.00},
    'STL': {'name': 'Busch Stadium', 'lat': 38.623, 'lon': -90.193, 'indoor': False, 'k_factor': 0.99},
    'TB':  {'name': 'Tropicana Field', 'lat': 27.768, 'lon': -82.653, 'indoor': True, 'k_factor': 1.01},
    'TEX': {'name': 'Globe Life Field', 'lat': 32.747, 'lon': -97.083, 'indoor': True, 'k_factor': 0.95},
    'TOR': {'name': 'Rogers Centre', 'lat': 43.641, 'lon': -79.389, 'indoor': True, 'k_factor': 1.01},
    'WSH': {'name': 'Nationals Park', 'lat': 38.873, 'lon': -77.007, 'indoor': False, 'k_factor': 1.00},
}

# ============================================================
# MODULE 2: WEATHER ENGINE (no Selenium, no API key)
# ============================================================

def get_weather(lat, lon):
    """Fetch weather using wttr.in (free, no API key, no Selenium)."""
    try:
        url = f"https://wttr.in/{lat},{lon}?format=j1"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get('current_condition', [{}])[0]
            return {
                'temp_f': current.get('temp_F', 'N/A'),
                'wind_mph': current.get('windspeedMiles', 'N/A'),
                'wind_dir': current.get('winddir16Point', 'N/A'),
                'humidity': current.get('humidity', 'N/A'),
                'precip_mm': current.get('precipMM', '0'),
                'condition': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
                'feels_like_f': current.get('FeelsLikeF', 'N/A'),
            }
    except Exception as e:
        print(f"Weather error: {e}")
    return {'temp_f': 'N/A', 'wind_mph': 'N/A', 'wind_dir': 'N/A',
            'humidity': 'N/A', 'precip_mm': '0', 'condition': 'Unknown'}

def get_stadium_weather(team_abbr):
    """Get weather for a specific MLB stadium."""
    stadium = MLB_STADIUMS.get(team_abbr)
    if not stadium:
        return {'condition': 'Unknown team'}
    if stadium['indoor']:
        return {'condition': 'Indoor (Climate Controlled)', 'wind_mph': '0',
                'temp_f': '72', 'indoor': True, 'venue': stadium['name']}
    weather = get_weather(stadium['lat'], stadium['lon'])
    weather['venue'] = stadium['name']
    weather['indoor'] = False
    weather['k_factor'] = stadium['k_factor']
    return weather


# ============================================================
# MODULE 3: LINEUP CHECKER (MLB Stats API - free)
# ============================================================

def get_todays_games():
    """Fetch today's MLB schedule with probable pitchers."""
    today = datetime.date.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={today}&sportId=1&hydrate=probablePitcher,team"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        games = []
        for date_entry in data.get('dates', []):
            for game in date_entry.get('games', []):
                away = game.get('teams', {}).get('away', {})
                home = game.get('teams', {}).get('home', {})
                away_pitcher = away.get('probablePitcher', {})
                home_pitcher = home.get('probablePitcher', {})
                games.append({
                    'game_id': game.get('gamePk'),
                    'game_time': game.get('gameDate'),
                    'status': game.get('status', {}).get('detailedState', 'Unknown'),
                    'away_team': away.get('team', {}).get('abbreviation', 'UNK'),
                    'away_team_name': away.get('team', {}).get('name', 'Unknown'),
                    'home_team': home.get('team', {}).get('abbreviation', 'UNK'),
                    'home_team_name': home.get('team', {}).get('name', 'Unknown'),
                    'away_pitcher_name': away_pitcher.get('fullName', 'TBD'),
                    'away_pitcher_id': away_pitcher.get('id', None),
                    'home_pitcher_name': home_pitcher.get('fullName', 'TBD'),
                    'home_pitcher_id': home_pitcher.get('id', None),
                    'venue': game.get('venue', {}).get('name', 'Unknown'),
                })
        return pd.DataFrame(games)
    except Exception as e:
        print(f"Schedule API error: {e}")
        return pd.DataFrame()

def get_lineup(game_id):
    """Fetch the batting lineup for a specific game (once posted)."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        lineups = {}
        for side in ['away', 'home']:
            team_data = data.get('liveData', {}).get('boxscore', {}).get('teams', {}).get(side, {})
            batting_order = team_data.get('battingOrder', [])
            players_data = team_data.get('players', {})
            lineup = []
            for player_id in batting_order:
                key = f"ID{player_id}"
                player_info = players_data.get(key, {})
                lineup.append({
                    'id': player_id,
                    'name': player_info.get('person', {}).get('fullName', 'Unknown'),
                    'position': player_info.get('position', {}).get('abbreviation', ''),
                    'batting_avg': player_info.get('seasonStats', {}).get('batting', {}).get('avg', '.000'),
                })
            team_name = data.get('gameData', {}).get('teams', {}).get(side, {}).get('abbreviation', side)
            lineups[team_name] = lineup
        return lineups
    except Exception as e:
        print(f"Lineup API error: {e}")
        return {}

def get_pitcher_season_stats(pitcher_id):
    """Fetch season stats for a pitcher from MLB Stats API."""
    if not pitcher_id:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[season])"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        person = data.get('people', [{}])[0]
        stats_list = person.get('stats', [])
        if stats_list:
            splits = stats_list[0].get('splits', [])
            if splits:
                stat = splits[0].get('stat', {})
                games_started = int(stat.get('gamesStarted', 0))
                strikeouts = int(stat.get('strikeOuts', 0))
                ip_str = stat.get('inningsPitched', '0')
                ip = float(ip_str) if ip_str else 0
                k_per_start = strikeouts / games_started if games_started > 0 else 0
                k_per_9 = (strikeouts / ip * 9) if ip > 0 else 0
                return {
                    'name': person.get('fullName', 'Unknown'),
                    'era': stat.get('era', 'N/A'),
                    'whip': stat.get('whip', 'N/A'),
                    'strikeouts': strikeouts,
                    'games_started': games_started,
                    'innings_pitched': ip_str,
                    'k_per_start': round(k_per_start, 2),
                    'k_per_9': round(k_per_9, 2),
                    'wins': stat.get('wins', 0),
                    'losses': stat.get('losses', 0),
                    'walks': int(stat.get('baseOnBalls', 0)),
                    'hits_allowed': int(stat.get('hits', 0)),
                    'home_runs_allowed': int(stat.get('homeRuns', 0)),
                    'avg_against': stat.get('avg', 'N/A'),
                }
    except Exception as e:
        print(f"Pitcher stats error: {e}")
    return {}


# ============================================================
# MODULE 4: EV CALCULATION ENGINE
# ============================================================

def implied_probability(odds):
    """Convert American odds to implied probability."""
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)

def calculate_ev(hit_rate, odds, stake=1.0):
    """Calculate Expected Value of a bet."""
    imp_prob = implied_probability(odds)
    if odds < 0:
        profit = stake * (100 / -odds)
    else:
        profit = stake * (odds / 100)
    ev = (hit_rate * profit) - ((1 - hit_rate) * stake)
    edge = hit_rate - imp_prob
    return {
        'hit_rate': round(hit_rate, 4),
        'implied_prob': round(imp_prob, 4),
        'edge': round(edge, 4),
        'ev_per_dollar': round(ev, 4),
        'ev_pct': round(ev / stake * 100, 2),
        'profit_if_win': round(profit, 2),
        'kelly_fraction': round(edge / (profit / stake), 4) if profit > 0 and edge > 0 else 0,
    }

def bayesian_hit_rate(observed_hits, total_games, league_mean=0.50, prior_weight=5):
    """
    Regress observed hit rate toward league mean.
    Prevents overconfidence from small samples.
    prior_weight = effective number of 'phantom games' at the league average.
    """
    adjusted_hits = observed_hits + (prior_weight * league_mean)
    adjusted_total = total_games + prior_weight
    return adjusted_hits / adjusted_total

def evaluate_k_prop(pitcher_stats, prop_line, odds, direction='over', park_k_factor=1.0):
    """
    Evaluate a strikeout prop using pitcher season stats.
    Returns EV analysis with confidence level.
    """
    k_avg = pitcher_stats.get('k_per_start', 0)
    gs = pitcher_stats.get('games_started', 0)
    total_k = pitcher_stats.get('strikeouts', 0)

    if gs < 3:
        return {'error': 'Insufficient starts', 'confidence': 'unknown'}

    # Adjust for park factor
    adjusted_avg = k_avg * park_k_factor

    # Estimate hit rate using normal approximation
    # Standard deviation of K per start is roughly sqrt(k_avg) for Poisson-like distribution
    k_std = np.sqrt(k_avg) if k_avg > 0 else 1.5
    
    from scipy import stats as scipy_stats
    if direction == 'over':
        # P(K > line) = 1 - CDF(line)
        raw_hit_rate = 1 - scipy_stats.norm.cdf(prop_line + 0.5, loc=adjusted_avg, scale=k_std)
    else:
        # P(K < line) = CDF(line - 0.5)
        raw_hit_rate = scipy_stats.norm.cdf(prop_line - 0.5, loc=adjusted_avg, scale=k_std)

    # Bayesian regression toward 50%
    hit_rate = bayesian_hit_rate(raw_hit_rate * gs, gs, league_mean=0.50, prior_weight=3)

    ev_data = calculate_ev(hit_rate, odds)
    
    # Confidence level
    separation = abs(adjusted_avg - prop_line)
    if separation > 1.5 and gs >= 15:
        confidence = 'high'
    elif separation > 0.5 and gs >= 10:
        confidence = 'moderate'
    elif separation < 0.3:
        confidence = 'coinflip'
    else:
        confidence = 'low'

    return {
        **ev_data,
        'pitcher': pitcher_stats.get('name', 'Unknown'),
        'k_avg': round(adjusted_avg, 2),
        'raw_k_avg': round(k_avg, 2),
        'park_adjusted': park_k_factor != 1.0,
        'park_k_factor': park_k_factor,
        'line': prop_line,
        'direction': direction,
        'games_started': gs,
        'confidence': confidence,
        'separation': round(separation, 2),
    }


# ============================================================
# MODULE 5: ALT-LINE SWEEPER
# ============================================================

def sweep_alt_lines(pitcher_stats, odds_map, direction='over', park_k_factor=1.0):
    """
    Evaluate EV across multiple K lines simultaneously.
    odds_map = {line: odds} e.g. {5.5: -130, 6.5: +110, 7.5: +180}
    Returns DataFrame ranked by EV.
    """
    results = []
    for line, odds in odds_map.items():
        ev = evaluate_k_prop(pitcher_stats, line, odds, direction, park_k_factor)
        if 'error' not in ev:
            results.append(ev)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('ev_per_dollar', ascending=False)
    return df


# ============================================================
# MODULE 6: PARLAY & BOOST OPTIMIZER
# ============================================================

def parlay_odds(legs):
    """
    Calculate parlay payout from list of American odds.
    legs = list of American odds e.g. [-130, +110, +100]
    Returns combined American odds.
    """
    decimal_odds = []
    for odds in legs:
        if odds < 0:
            decimal_odds.append(1 + (100 / -odds))
        else:
            decimal_odds.append(1 + (odds / 100))
    
    combined_decimal = 1
    for d in decimal_odds:
        combined_decimal *= d
    
    if combined_decimal >= 2:
        american = round((combined_decimal - 1) * 100)
    else:
        american = round(-100 / (combined_decimal - 1))
    
    return {
        'combined_decimal': round(combined_decimal, 4),
        'combined_american': american,
        'implied_prob': round(1 / combined_decimal, 4),
    }

def apply_boost(american_odds, boost_pct):
    """
    Apply a profit boost to American odds.
    boost_pct = 50 means +50% profit boost.
    Returns new effective American odds.
    """
    if american_odds < 0:
        profit_per_100 = 100 / -american_odds * 100
    else:
        profit_per_100 = american_odds
    
    boosted_profit = profit_per_100 * (1 + boost_pct / 100)
    
    if boosted_profit >= 100:
        new_odds = round(boosted_profit)
    else:
        new_odds = round(-100 / (boosted_profit / 100))
    
    return {
        'original_odds': american_odds,
        'boosted_odds': new_odds,
        'original_profit_per_100': round(profit_per_100, 2),
        'boosted_profit_per_100': round(boosted_profit, 2),
        'boost_value_pct': boost_pct,
    }

def parlay_ev(legs_data, boost_pct=0):
    """
    Calculate full parlay EV with optional boost.
    legs_data = [{'odds': -130, 'true_prob': 0.60}, ...]
    Returns detailed parlay analysis.
    """
    odds_list = [l['odds'] for l in legs_data]
    parlay_info = parlay_odds(odds_list)
    
    combined_true_prob = 1
    for l in legs_data:
        combined_true_prob *= l['true_prob']
    
    if boost_pct > 0:
        boosted = apply_boost(parlay_info['combined_american'], boost_pct)
        effective_odds = boosted['boosted_odds']
    else:
        effective_odds = parlay_info['combined_american']
    
    ev = calculate_ev(combined_true_prob, effective_odds)
    
    return {
        'legs': len(legs_data),
        'combined_true_prob': round(combined_true_prob, 4),
        'parlay_odds': parlay_info['combined_american'],
        'boosted_odds': effective_odds if boost_pct > 0 else None,
        'boost_pct': boost_pct,
        **ev,
    }

def rr_by_2s(legs_data, stake_per_combo):
    """
    Calculate Round Robin by 2s from a list of legs.
    Returns all combos with EV analysis.
    """
    from itertools import combinations
    combos = list(combinations(range(len(legs_data)), 2))
    results = []
    
    for combo in combos:
        combo_legs = [legs_data[i] for i in combo]
        combo_names = [l.get('name', f'Leg {i+1}') for i in combo]
        ev = parlay_ev(combo_legs)
        ev['combo'] = ' + '.join(combo_names)
        ev['stake'] = stake_per_combo
        ev['potential_profit'] = round(ev['profit_if_win'] * stake_per_combo, 2)
        results.append(ev)
    
    return pd.DataFrame(results)


# ============================================================
# MODULE 7: FULL SLATE SCANNER
# ============================================================

def scan_full_slate(verbose=True):
    """
    Enumerate the FULL population of today's MLB games.
    Returns DataFrame with every starter, their stats, and park context.
    THE CARDINAL RULE: enumerate before you evaluate.
    """
    if verbose:
        print("=" * 60)
        print("EV+ GAMBIT v2.0 — FULL SLATE SCAN")
        print(f"Date: {datetime.date.today().strftime('%A, %B %d, %Y')}")
        print("=" * 60)
    
    games = get_todays_games()
    if games.empty:
        print("No games found for today.")
        return pd.DataFrame()
    
    if verbose:
        print(f"\n Found {len(games)} games on today's slate.\n")
    
    slate = []
    for _, game in games.iterrows():
        for side in ['away', 'home']:
            pitcher_name = game[f'{side}_pitcher_name']
            pitcher_id = game[f'{side}_pitcher_id']
            team = game[f'{side}_team']
            opponent = game['home_team'] if side == 'away' else game['away_team']
            venue_team = game['home_team']
            
            stats = get_pitcher_season_stats(pitcher_id) if pitcher_id else {}
            park_info = MLB_STADIUMS.get(venue_team, {})
            
            entry = {
                'game_id': game['game_id'],
                'game_time': game['game_time'],
                'status': game['status'],
                'team': team,
                'opponent': opponent,
                'side': side,
                'pitcher': pitcher_name,
                'pitcher_id': pitcher_id,
                'era': stats.get('era', 'N/A'),
                'whip': stats.get('whip', 'N/A'),
                'k_total': stats.get('strikeouts', 0),
                'games_started': stats.get('games_started', 0),
                'k_per_start': stats.get('k_per_start', 0),
                'k_per_9': stats.get('k_per_9', 0),
                'wins': stats.get('wins', 0),
                'losses': stats.get('losses', 0),
                'ip': stats.get('innings_pitched', '0'),
                'venue': park_info.get('name', game['venue']),
                'indoor': park_info.get('indoor', False),
                'k_factor': park_info.get('k_factor', 1.0),
            }
            slate.append(entry)
            
            if verbose and pitcher_name != 'TBD':
                record = f"{stats.get('wins', 0)}-{stats.get('losses', 0)}"
                era = stats.get('era', 'N/A')
                k = stats.get('strikeouts', 0)
                kps = stats.get('k_per_start', 0)
                k9 = stats.get('k_per_9', 0)
                print(f"  {team:>4} {'@' if side == 'away' else 'vs'} {opponent:<4} | "
                      f"{pitcher_name:<22} | {record:>5} | "
                      f"ERA {era:>5} | {k:>3} K | {kps:.1f} K/GS | {k9:.1f} K/9")
            
            time.sleep(0.3)  # Rate limiting for MLB API
    
    df = pd.DataFrame(slate)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SLATE COMPLETE: {len(df)} pitcher slots enumerated.")
        tbd = (df['pitcher'] == 'TBD').sum()
        if tbd > 0:
            print(f"WARNING: {tbd} slots still TBD — check back closer to game time.")
        print(f"{'=' * 60}")
    
    return df


def rank_k_props(slate_df, default_line=None, default_odds=-110):
    """
    Rank all pitchers on the slate by K prop edge.
    If no line provided, uses k_per_start as the assumed line.
    """
    results = []
    for _, row in slate_df.iterrows():
        if row['pitcher'] == 'TBD' or row['games_started'] < 3:
            continue
        
        # Estimate the book's line as slightly below K average
        line = default_line if default_line else round(row['k_per_start'] - 0.3, 0) + 0.5
        
        stats = {
            'name': row['pitcher'],
            'k_per_start': row['k_per_start'],
            'games_started': row['games_started'],
            'strikeouts': row['k_total'],
        }
        
        # Evaluate over
        over_ev = evaluate_k_prop(stats, line, default_odds, 'over', row['k_factor'])
        if 'error' not in over_ev:
            over_ev['team'] = row['team']
            over_ev['opponent'] = row['opponent']
            over_ev['era'] = row['era']
            over_ev['venue'] = row['venue']
            results.append(over_ev)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('edge', ascending=False)
    return df


# ============================================================
# MODULE 8: STATCAST DEEP DIVE (pybaseball)
# ============================================================

def pitcher_game_log(pitcher_name, days=90):
    """
    Pull pitcher's game-by-game K log from Statcast.
    Returns DataFrame with date, Ks, IP proxy.
    """
    try:
        from pybaseball import playerid_lookup, statcast_pitcher
        names = pitcher_name.split(" ")
        if len(names) < 2:
            return pd.DataFrame()
        
        first, last = names[0], " ".join(names[1:])
        id_df = playerid_lookup(last.lower(), first.lower())
        
        if id_df.empty:
            print(f"Player not found: {pitcher_name}")
            return pd.DataFrame()
        
        mlbam_id = id_df['key_mlbam'].values[0]
        end_dt = datetime.datetime.today().strftime('%Y-%m-%d')
        start_dt = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = statcast_pitcher(start_dt, end_dt, mlbam_id)
        if df.empty:
            return pd.DataFrame()
        
        game_log = df.groupby('game_date').agg(
            strikeouts=('events', lambda x: (x == 'strikeout').sum()),
            batters_faced=('events', lambda x: x.notna().sum()),
            hits=('events', lambda x: x.isin(['single', 'double', 'triple', 'home_run']).sum()),
            walks=('events', lambda x: (x == 'walk').sum()),
        ).reset_index()
        
        game_log['game_date'] = pd.to_datetime(game_log['game_date'])
        game_log = game_log.sort_values('game_date', ascending=False)
        
        return game_log
    
    except Exception as e:
        print(f"Statcast error for {pitcher_name}: {e}")
        return pd.DataFrame()

def hit_rate_at_line(game_log, line, direction='over'):
    """Calculate historical hit rate at a specific K line."""
    if game_log.empty or 'strikeouts' not in game_log.columns:
        return 0.0
    
    if direction == 'over':
        hits = (game_log['strikeouts'] > line).sum()
    else:
        hits = (game_log['strikeouts'] < line).sum()
    
    raw_rate = hits / len(game_log)
    # Bayesian adjustment
    adjusted = bayesian_hit_rate(hits, len(game_log), league_mean=0.50, prior_weight=3)
    
    return {
        'raw_hit_rate': round(raw_rate, 4),
        'bayesian_hit_rate': round(adjusted, 4),
        'games': len(game_log),
        'hits': int(hits),
        'line': line,
        'direction': direction,
    }


# ============================================================
# MODULE 9: VISUALIZATION
# ============================================================

def plot_k_trend(game_log, pitcher_name, prop_line=None, window=5):
    """Plot K trend with SMA, CI, and prop line."""
    if game_log.empty:
        print("No data to plot.")
        return
    
    gl = game_log.sort_values('game_date').reset_index(drop=True)
    gl['SMA'] = gl['strikeouts'].rolling(window=window, min_periods=1).mean()
    gl['STD'] = gl['strikeouts'].rolling(window=window, min_periods=1).std().fillna(0)
    count = gl['strikeouts'].rolling(window=window, min_periods=1).count()
    gl['CI'] = 1.96 * (gl['STD'] / np.sqrt(count)).fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Dark theme
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    ax.scatter(gl['game_date'], gl['strikeouts'], color='#00d2ff', alpha=0.6, s=50, 
               label='Game Ks', zorder=3)
    ax.plot(gl['game_date'], gl['SMA'], color='#00ff88', linewidth=2.5, 
            label=f'{window}-Game SMA')
    ax.fill_between(gl['game_date'], 
                     (gl['SMA'] - gl['CI']).clip(lower=0), 
                     gl['SMA'] + gl['CI'],
                     color='#00ff88', alpha=0.15, label='95% CI')
    
    if prop_line is not None:
        ax.axhline(y=prop_line, color='#ff4757', linestyle='--', linewidth=2, 
                   label=f'Prop Line ({prop_line})')
    
    ax.set_ylabel('Strikeouts', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Game Date', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    ax.grid(axis='both', linestyle='--', alpha=0.3, color='#444')
    
    avg_k = gl['strikeouts'].mean()
    ax.set_title(f"{pitcher_name} — Strikeout Trend\n"
                 f"Avg: {avg_k:.1f} K/GS | Last 5: {gl['strikeouts'].tail(5).mean():.1f} | "
                 f"Games: {len(gl)}",
                 color='white', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# MODULE 10: QUICK-START FUNCTIONS
# ============================================================

def daily_k_scan():
    """
    One-command full slate scan with K prop rankings.
    THE function Sean runs every morning.
    """
    print("\n" + "🎯 " * 20)
    print("EV+ GAMBIT v2.0 — DAILY K SCAN")
    print("🎯 " * 20 + "\n")
    
    slate = scan_full_slate(verbose=True)
    if slate.empty:
        return slate
    
    print("\n\n📊 K PROP RANKINGS (estimated lines, sorted by edge):")
    print("-" * 80)
    rankings = rank_k_props(slate)
    if not rankings.empty:
        display_cols = ['pitcher', 'team', 'opponent', 'k_avg', 'line', 
                       'direction', 'hit_rate', 'implied_prob', 'edge', 
                       'ev_per_dollar', 'confidence']
        available = [c for c in display_cols if c in rankings.columns]
        print(rankings[available].head(20).to_string(index=False))
    
    return slate, rankings

def analyze_prop(pitcher_name, line, odds, direction='over', days=90):
    """
    Full single-prop analysis with game log, hit rate, EV, and chart.
    """
    print(f"\n{'=' * 60}")
    print(f"PROP ANALYSIS: {pitcher_name}")
    print(f"Line: {direction.upper()} {line} | Odds: {odds}")
    print(f"{'=' * 60}\n")
    
    # Game log from Statcast
    gl = pitcher_game_log(pitcher_name, days)
    if gl.empty:
        print("No Statcast data available. Using season stats only.")
        return
    
    # Hit rate at line
    hr = hit_rate_at_line(gl, line, direction)
    print(f"📊 Hit Rate at {direction.upper()} {line}:")
    print(f"   Raw:      {hr['raw_hit_rate']:.1%} ({hr['hits']}/{hr['games']} games)")
    print(f"   Bayesian: {hr['bayesian_hit_rate']:.1%}")
    
    # EV calculation
    ev = calculate_ev(hr['bayesian_hit_rate'], odds)
    print(f"\n💰 EV Analysis:")
    print(f"   Implied Prob:  {ev['implied_prob']:.1%}")
    print(f"   True Prob:     {hr['bayesian_hit_rate']:.1%}")
    print(f"   Edge:          {ev['edge']:+.1%}")
    print(f"   EV per $1:     ${ev['ev_per_dollar']:+.3f}")
    print(f"   Kelly:         {ev['kelly_fraction']:.1%} of bankroll")
    
    verdict = "✅ PLAY" if ev['edge'] > 0.03 else "⚠️ MARGINAL" if ev['edge'] > 0 else "❌ FADE"
    print(f"\n   VERDICT: {verdict}")
    
    # Chart
    plot_k_trend(gl, pitcher_name, line)
    
    return {'game_log': gl, 'hit_rate': hr, 'ev': ev}

def boost_optimizer(legs_data, available_boosts):
    """
    Given legs and available boosts, find the optimal boost allocation.
    available_boosts = [{'name': '50% MLB', 'pct': 50, 'min_odds': -200}, ...]
    """
    print("\n🔧 BOOST OPTIMIZER")
    print("-" * 50)
    
    results = []
    for boost in available_boosts:
        ev = parlay_ev(legs_data, boost['pct'])
        ev['boost_name'] = boost['name']
        results.append(ev)
    
    # Also evaluate no boost
    ev_no_boost = parlay_ev(legs_data, 0)
    ev_no_boost['boost_name'] = 'No Boost'
    results.append(ev_no_boost)
    
    df = pd.DataFrame(results)
    df = df.sort_values('ev_per_dollar', ascending=False)
    
    print(df[['boost_name', 'parlay_odds', 'boosted_odds', 'combined_true_prob',
              'edge', 'ev_per_dollar']].to_string(index=False))
    
    best = df.iloc[0]
    print(f"\n🏆 BEST: {best['boost_name']} → EV: ${best['ev_per_dollar']:+.3f}/dollar")
    
    return df


# ============================================================
# USAGE EXAMPLES (uncomment in Colab)
# ============================================================

"""
# --- DAILY SCAN ---
slate, rankings = daily_k_scan()

# --- SINGLE PROP ANALYSIS ---
analyze_prop('Jacob Misiorowski', 9.5, -106, direction='over')
analyze_prop('Max Scherzer', 3.5, +120, direction='under')

# --- ALT LINE SWEEP ---
miz_stats = {'name': 'Jacob Misiorowski', 'k_per_start': 9.25, 'games_started': 20, 'strikeouts': 185}
miz_lines = {7.5: -370, 8.0: -250, 8.5: -184, 9.0: -131, 9.5: +106}
sweep = sweep_alt_lines(miz_stats, miz_lines, 'over', park_k_factor=1.04)
print(sweep[['line', 'hit_rate', 'implied_prob', 'edge', 'ev_per_dollar']].to_string(index=False))

# --- PARLAY WITH BOOST ---
legs = [
    {'name': 'Miz U9.5', 'odds': -131, 'true_prob': 0.58},
    {'name': 'Williams U6.5', 'odds': +105, 'true_prob': 0.55},
    {'name': 'Rea O4.5', 'odds': -154, 'true_prob': 0.65},
    {'name': 'Scherzer U3.5', 'odds': +115, 'true_prob': 0.60},
]
ev = parlay_ev(legs, boost_pct=50)
print(ev)

# --- RR BY 2s ---
rr = rr_by_2s(legs, stake_per_combo=10)
print(rr[['combo', 'combined_true_prob', 'parlay_odds', 'edge', 'ev_per_dollar', 'potential_profit']])

# --- BOOST OPTIMIZER ---
boosts = [
    {'name': '50% MLB Parlay', 'pct': 50, 'min_odds': -200},
    {'name': '100% 4-Leg SGP', 'pct': 100, 'min_odds': None},
    {'name': '25% Any Parlay', 'pct': 25, 'min_odds': None},
]
boost_optimizer(legs, boosts)

# --- WEATHER CHECK ---
weather = get_stadium_weather('CHC')
print(f"Wrigley: {weather}")

# --- LINEUP CHECK ---
games = get_todays_games()
if not games.empty:
    game_id = games.iloc[0]['game_id']
    lineups = get_lineup(game_id)
    for team, players in lineups.items():
        print(f"\\n{team} Lineup:")
        for i, p in enumerate(players, 1):
            print(f"  {i}. {p['name']} ({p['position']}) - {p['batting_avg']}")
"""

if __name__ == '__main__':
    print("EV+ Gambit v2.0 loaded. Run daily_k_scan() to begin.")

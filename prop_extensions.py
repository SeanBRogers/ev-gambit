"""
EV+ GAMBIT v2.1 — Extended Prop Module
Adds evaluation for ALL DraftKings MLB prop types:
- Batter: Hits, Total Bases, Home Runs, RBIs, Runs, Stolen Bases
- Pitcher: Hits Allowed, Walks Allowed, Earned Runs, Outs Recorded
- Game: NRFI/YRFI, Totals, ML
Drop this in the same directory as ev_gambit_v2.py
"""

import requests
import numpy as np
import pandas as pd
import datetime
import time

from ev_gambit_v2 import (
    implied_probability, calculate_ev, bayesian_hit_rate,
    MLB_STADIUMS, get_todays_games, evaluate_k_prop, PREFERRED_BOOKS
)


# ============================================================
# BATTER STATS FROM MLB STATS API (free, no key)
# ============================================================

def get_batter_season_stats(player_id):
    """Fetch full season batting stats from MLB Stats API."""
    if not player_id:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=[hitting],type=[season])"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        person = data.get('people', [{}])[0]
        stats_list = person.get('stats', [])
        if stats_list:
            splits = stats_list[0].get('splits', [])
            if splits:
                s = splits[0].get('stat', {})
                gp = int(s.get('gamesPlayed', 0))
                if gp == 0:
                    return {}
                return {
                    'name': person.get('fullName', 'Unknown'),
                    'games_played': gp,
                    'avg': s.get('avg', '.000'),
                    'obp': s.get('obp', '.000'),
                    'slg': s.get('slg', '.000'),
                    'ops': s.get('ops', '.000'),
                    'hits': int(s.get('hits', 0)),
                    'doubles': int(s.get('doubles', 0)),
                    'triples': int(s.get('triples', 0)),
                    'home_runs': int(s.get('homeRuns', 0)),
                    'rbi': int(s.get('rbi', 0)),
                    'runs': int(s.get('runs', 0)),
                    'stolen_bases': int(s.get('stolenBases', 0)),
                    'total_bases': int(s.get('totalBases', 0)),
                    'walks': int(s.get('baseOnBalls', 0)),
                    'strikeouts': int(s.get('strikeOuts', 0)),
                    'plate_appearances': int(s.get('plateAppearances', 0)),
                    'at_bats': int(s.get('atBats', 0)),
                    # Per-game rates
                    'hits_pg': round(int(s.get('hits', 0)) / gp, 3),
                    'tb_pg': round(int(s.get('totalBases', 0)) / gp, 3),
                    'hr_pg': round(int(s.get('homeRuns', 0)) / gp, 3),
                    'rbi_pg': round(int(s.get('rbi', 0)) / gp, 3),
                    'runs_pg': round(int(s.get('runs', 0)) / gp, 3),
                    'sb_pg': round(int(s.get('stolenBases', 0)) / gp, 3),
                    'bb_pg': round(int(s.get('baseOnBalls', 0)) / gp, 3),
                    'k_pg': round(int(s.get('strikeOuts', 0)) / gp, 3),
                }
    except Exception as e:
        print(f"Batter stats error: {e}")
    return {}


def get_pitcher_extended_stats(pitcher_id):
    """Fetch extended pitching stats for hits/walks/ER allowed props."""
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
                s = splits[0].get('stat', {})
                gs = int(s.get('gamesStarted', 0))
                if gs == 0:
                    return {}
                ip_str = s.get('inningsPitched', '0')
                ip = float(ip_str) if ip_str else 0
                outs = int(ip * 3 + (float(ip_str.split('.')[-1]) if '.' in ip_str else 0)) if ip_str else 0
                return {
                    'name': person.get('fullName', 'Unknown'),
                    'games_started': gs,
                    'innings_pitched': ip,
                    'hits_allowed': int(s.get('hits', 0)),
                    'walks_allowed': int(s.get('baseOnBalls', 0)),
                    'earned_runs': int(s.get('earnedRuns', 0)),
                    'strikeouts': int(s.get('strikeOuts', 0)),
                    'home_runs_allowed': int(s.get('homeRuns', 0)),
                    # Per-start rates
                    'hits_ps': round(int(s.get('hits', 0)) / gs, 2),
                    'walks_ps': round(int(s.get('baseOnBalls', 0)) / gs, 2),
                    'er_ps': round(int(s.get('earnedRuns', 0)) / gs, 2),
                    'k_ps': round(int(s.get('strikeOuts', 0)) / gs, 2),
                    'outs_ps': round((ip * 3) / gs, 2) if gs > 0 else 0,
                    'ip_ps': round(ip / gs, 2),
                }
    except Exception as e:
        print(f"Pitcher extended stats error: {e}")
    return {}


# ============================================================
# UNIVERSAL PROP EVALUATOR
# ============================================================

def evaluate_prop(stat_avg, stat_std, prop_line, odds, direction='over',
                  games_played=50, stat_name='stat'):
    """
    Universal prop evaluator. Works for any stat with a per-game average.
    
    stat_avg: player's per-game average for this stat
    stat_std: standard deviation (estimated from Poisson or provided)
    prop_line: the book's line (e.g., 1.5 for O/U 1.5 hits)
    odds: American odds
    direction: 'over' or 'under'
    games_played: sample size for Bayesian adjustment
    """
    from scipy import stats as scipy_stats

    if direction == 'over':
        raw_hit_rate = 1 - scipy_stats.norm.cdf(prop_line + 0.5, loc=stat_avg, scale=stat_std)
    else:
        raw_hit_rate = scipy_stats.norm.cdf(prop_line - 0.5, loc=stat_avg, scale=stat_std)

    hit_rate = bayesian_hit_rate(raw_hit_rate * games_played, games_played,
                                 league_mean=0.50, prior_weight=3)

    ev_data = calculate_ev(hit_rate, odds)

    separation = abs(stat_avg - prop_line)
    if separation > 0.5 and games_played >= 50:
        confidence = 'high'
    elif separation > 0.2 and games_played >= 30:
        confidence = 'moderate'
    elif separation < 0.1:
        confidence = 'coinflip'
    else:
        confidence = 'low'

    return {
        **ev_data,
        'stat_name': stat_name,
        'stat_avg': round(stat_avg, 3),
        'stat_std': round(stat_std, 3),
        'line': prop_line,
        'direction': direction,
        'separation': round(separation, 3),
        'confidence': confidence,
        'games': games_played,
    }


# ============================================================
# BATTER PROP EVALUATORS
# ============================================================

def evaluate_hits_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter hits prop (O/U 0.5, 1.5, 2.5)."""
    avg = batter_stats.get('hits_pg', 0)
    gp = batter_stats.get('games_played', 0)
    std = np.sqrt(avg) if avg > 0 else 0.5
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, 'hits')
    result['player'] = batter_stats.get('name', 'Unknown')
    result['batting_avg'] = batter_stats.get('avg', '.000')
    return result


def evaluate_tb_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter total bases prop."""
    avg = batter_stats.get('tb_pg', 0)
    gp = batter_stats.get('games_played', 0)
    std = np.sqrt(avg) if avg > 0 else 0.8
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, 'total_bases')
    result['player'] = batter_stats.get('name', 'Unknown')
    result['slg'] = batter_stats.get('slg', '.000')
    return result


def evaluate_hr_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter home run prop (usually O/U 0.5)."""
    avg = batter_stats.get('hr_pg', 0)
    gp = batter_stats.get('games_played', 0)
    # HR distribution is heavily right-skewed, use Poisson
    from scipy import stats as scipy_stats
    if direction == 'over':
        raw_hit_rate = 1 - scipy_stats.poisson.cdf(int(prop_line), avg)
    else:
        raw_hit_rate = scipy_stats.poisson.cdf(int(prop_line) - 1, avg) if prop_line >= 1 else 1 - avg
    
    hit_rate = bayesian_hit_rate(raw_hit_rate * gp, gp, league_mean=0.50, prior_weight=5)
    ev_data = calculate_ev(hit_rate, odds)
    
    return {
        **ev_data,
        'stat_name': 'home_runs',
        'player': batter_stats.get('name', 'Unknown'),
        'hr_pg': round(avg, 4),
        'total_hr': batter_stats.get('home_runs', 0),
        'line': prop_line,
        'direction': direction,
        'games': gp,
        'confidence': 'low' if gp < 50 else 'moderate',
    }


def evaluate_rbi_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter RBI prop."""
    avg = batter_stats.get('rbi_pg', 0)
    gp = batter_stats.get('games_played', 0)
    std = np.sqrt(avg) if avg > 0 else 0.5
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, 'rbi')
    result['player'] = batter_stats.get('name', 'Unknown')
    return result


def evaluate_runs_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter runs scored prop."""
    avg = batter_stats.get('runs_pg', 0)
    gp = batter_stats.get('games_played', 0)
    std = np.sqrt(avg) if avg > 0 else 0.5
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, 'runs')
    result['player'] = batter_stats.get('name', 'Unknown')
    return result


def evaluate_sb_prop(batter_stats, prop_line, odds, direction='over'):
    """Evaluate a batter stolen bases prop."""
    avg = batter_stats.get('sb_pg', 0)
    gp = batter_stats.get('games_played', 0)
    from scipy import stats as scipy_stats
    if direction == 'over':
        raw_hit_rate = 1 - scipy_stats.poisson.cdf(int(prop_line), avg)
    else:
        raw_hit_rate = scipy_stats.poisson.cdf(int(prop_line) - 1, avg) if prop_line >= 1 else 1 - avg
    
    hit_rate = bayesian_hit_rate(raw_hit_rate * gp, gp, league_mean=0.50, prior_weight=5)
    ev_data = calculate_ev(hit_rate, odds)
    
    return {
        **ev_data,
        'stat_name': 'stolen_bases',
        'player': batter_stats.get('name', 'Unknown'),
        'sb_pg': round(avg, 4),
        'total_sb': batter_stats.get('stolen_bases', 0),
        'line': prop_line,
        'direction': direction,
        'games': gp,
        'confidence': 'low',
    }


# ============================================================
# PITCHER EXTENDED PROP EVALUATORS
# ============================================================

def evaluate_hits_allowed_prop(pitcher_stats, prop_line, odds, direction='over'):
    """Evaluate pitcher hits allowed prop."""
    avg = pitcher_stats.get('hits_ps', 0)
    gs = pitcher_stats.get('games_started', 0)
    std = np.sqrt(avg) if avg > 0 else 1.5
    result = evaluate_prop(avg, std, prop_line, odds, direction, gs, 'hits_allowed')
    result['pitcher'] = pitcher_stats.get('name', 'Unknown')
    return result


def evaluate_walks_allowed_prop(pitcher_stats, prop_line, odds, direction='over'):
    """Evaluate pitcher walks allowed prop."""
    avg = pitcher_stats.get('walks_ps', 0)
    gs = pitcher_stats.get('games_started', 0)
    std = np.sqrt(avg) if avg > 0 else 0.8
    result = evaluate_prop(avg, std, prop_line, odds, direction, gs, 'walks_allowed')
    result['pitcher'] = pitcher_stats.get('name', 'Unknown')
    return result


def evaluate_er_prop(pitcher_stats, prop_line, odds, direction='over'):
    """Evaluate pitcher earned runs allowed prop."""
    avg = pitcher_stats.get('er_ps', 0)
    gs = pitcher_stats.get('games_started', 0)
    std = np.sqrt(avg) if avg > 0 else 0.8
    result = evaluate_prop(avg, std, prop_line, odds, direction, gs, 'earned_runs')
    result['pitcher'] = pitcher_stats.get('name', 'Unknown')
    return result


def evaluate_outs_prop(pitcher_stats, prop_line, odds, direction='over'):
    """Evaluate pitcher outs recorded prop."""
    avg = pitcher_stats.get('outs_ps', 0)
    gs = pitcher_stats.get('games_started', 0)
    std = np.sqrt(avg / 3) * 3 if avg > 0 else 2.0
    result = evaluate_prop(avg, std, prop_line, odds, direction, gs, 'outs_recorded')
    result['pitcher'] = pitcher_stats.get('name', 'Unknown')
    result['ip_ps'] = pitcher_stats.get('ip_ps', 0)
    return result


# ============================================================
# GAME-LEVEL PROP EVALUATORS
# ============================================================

def evaluate_nrfi(away_pitcher_stats, home_pitcher_stats, 
                  away_team_first_inning_runs_pg=0.5,
                  home_team_first_inning_runs_pg=0.5,
                  odds=-110):
    """
    Evaluate No Runs First Inning prop.
    Uses pitcher WHIP and ERA as proxies for first-inning scoring probability.
    """
    # Estimate first-inning scoreless probability for each half
    away_p_whip = float(away_pitcher_stats.get('whip', '1.30') or '1.30')
    home_p_whip = float(home_pitcher_stats.get('whip', '1.30') or '1.30')

    # Lower WHIP = fewer baserunners = higher scoreless probability
    # League average WHIP ~1.25, average NRFI rate ~48%
    away_scoreless_prob = max(0.30, min(0.80, 0.65 - (away_p_whip - 1.25) * 0.25))
    home_scoreless_prob = max(0.30, min(0.80, 0.65 - (home_p_whip - 1.25) * 0.25))

    nrfi_prob = away_scoreless_prob * home_scoreless_prob
    yrfi_prob = 1 - nrfi_prob

    nrfi_ev = calculate_ev(nrfi_prob, odds)
    yrfi_ev = calculate_ev(yrfi_prob, odds)

    return {
        'nrfi_prob': round(nrfi_prob, 4),
        'yrfi_prob': round(yrfi_prob, 4),
        'nrfi_ev': nrfi_ev,
        'yrfi_ev': yrfi_ev,
        'away_pitcher': away_pitcher_stats.get('name', 'Unknown'),
        'home_pitcher': home_pitcher_stats.get('name', 'Unknown'),
        'away_whip': away_p_whip,
        'home_whip': home_p_whip,
        'recommendation': 'NRFI' if nrfi_ev['edge'] > yrfi_ev['edge'] else 'YRFI',
    }


def evaluate_game_total(away_pitcher_era, home_pitcher_era, 
                        park_runs_factor=1.0, total_line=8.5, odds=-110,
                        direction='over'):
    """
    Evaluate game total (O/U) using pitcher ERAs and park factor.
    Simplified model: projected runs = (away_ERA + home_ERA) / 2 * park_factor * (9/9)
    """
    # Each pitcher's run contribution per 9 innings
    away_runs = float(away_pitcher_era) / 2  # rough: ERA/2 for 4.5 innings
    home_runs = float(home_pitcher_era) / 2
    
    # Bullpen adjustment: add ~1.5 runs for bullpen innings
    projected_total = (away_runs + home_runs + 3.0) * park_runs_factor
    
    std = 2.5  # Game totals have high variance
    
    from scipy import stats as scipy_stats
    if direction == 'over':
        hit_rate = 1 - scipy_stats.norm.cdf(total_line, loc=projected_total, scale=std)
    else:
        hit_rate = scipy_stats.norm.cdf(total_line, loc=projected_total, scale=std)
    
    ev_data = calculate_ev(hit_rate, odds)
    
    return {
        **ev_data,
        'projected_total': round(projected_total, 2),
        'line': total_line,
        'direction': direction,
        'park_factor': park_runs_factor,
        'confidence': 'moderate' if abs(projected_total - total_line) > 1 else 'low',
    }


# ============================================================
# MULTI-PROP SCANNER
# ============================================================

# ============================================================
# ODDS API TEAM NAME → ABBREVIATION MAPPING
# The Odds API uses full team names. MLB_STADIUMS uses abbreviations.
# This bridge ensures park factors come from the VENUE (home team
# per the Odds API), not from stale roster data.
# ============================================================

TEAM_NAME_TO_ABBR = {
    'Arizona Diamondbacks': 'AZ',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'ATH',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH',
}


def get_park_factor(home_team_abbr, factor_type='k_factor'):
    """Get park factor from home team abbreviation. Defaults to 1.0 if unknown."""
    stadium = MLB_STADIUMS.get(home_team_abbr, {})
    return stadium.get(factor_type, 1.0)


PROP_TYPE_MAP = {
    'pitcher_strikeouts': 'k',
    'batter_hits': 'hits',
    'batter_total_bases': 'tb',
    'batter_home_runs': 'hr',
    'batter_rbis': 'rbi',
    'batter_runs_scored': 'runs',
    'batter_stolen_bases': 'sb',
    'pitcher_hits_allowed': 'hits_allowed',
    'pitcher_walks': 'walks_allowed',
    'pitcher_earned_runs': 'earned_runs',
    'pitcher_outs': 'outs_recorded',
}

EVAL_FUNCTIONS = {
    'hits': evaluate_hits_prop,
    'tb': evaluate_tb_prop,
    'hr': evaluate_hr_prop,
    'rbi': evaluate_rbi_prop,
    'runs': evaluate_runs_prop,
    'sb': evaluate_sb_prop,
    'hits_allowed': evaluate_hits_allowed_prop,
    'walks_allowed': evaluate_walks_allowed_prop,
    'earned_runs': evaluate_er_prop,
    'outs_recorded': evaluate_outs_prop,
}


def fetch_all_props(api_key, sport='baseball_mlb'):
    """Fetch ALL prop types from The Odds API."""
    markets = ','.join([
        'pitcher_strikeouts',
        'batter_hits', 'batter_total_bases', 'batter_home_runs',
        'batter_rbis', 'batter_runs_scored', 'batter_stolen_bases',
        'pitcher_hits_allowed', 'pitcher_walks', 'pitcher_earned_runs',
        'pitcher_outs'
    ])
    
    events_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events'
    try:
        events_res = requests.get(events_url, params={'apiKey': api_key})
        events = events_res.json()
    except Exception as e:
        print(f"Events API error: {e}")
        return pd.DataFrame()
    
    odds_data = []
    for i, event in enumerate(events[:15]):
        event_id = event.get('id')
        if not event_id:
            continue
        
        home_team_full = event.get('home_team', '')
        away_team_full = event.get('away_team', '')
        home_team_abbr = TEAM_NAME_TO_ABBR.get(home_team_full, '')
        
        odds_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds'
        odds_params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american',
            'bookmakers': PREFERRED_BOOKS
        }
        try:
            odds_res = requests.get(odds_url, params=odds_params)
            if odds_res.status_code == 200:
                for bookmaker in odds_res.json().get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'event': f"{away_team_full} @ {home_team_full}",
                                'home_team': home_team_full,
                                'home_team_abbr': home_team_abbr,
                                'game_time': event.get('commence_time', ''),
                                'player': outcome.get('description', outcome.get('name', 'Unknown')),
                                'prop_type': market.get('key', ''),
                                'prop_category': PROP_TYPE_MAP.get(market.get('key', ''), 'unknown'),
                                'line': outcome.get('point', 0.0),
                                'odds': outcome.get('price', 0),
                                'direction': outcome.get('name', '').lower(),
                                'book': bookmaker.get('key', 'unknown'),
                            })
        except:
            continue
        time.sleep(0.5)
    
    return pd.DataFrame(odds_data)


def scan_all_props(api_key):
    """
    THE BIG SCAN. Pull every prop from every game, evaluate each one,
    and return a ranked DataFrame of edges.
    
    CRITICAL: Park factor comes from the Odds API event's HOME TEAM,
    not from player roster data. This ensures post-deadline trades
    don't corrupt the venue assignment.
    """
    print("=" * 60)
    print("EV+ GAMBIT v2.2 — FULL PROP SCANNER")
    print(f"Date: {datetime.date.today().strftime('%A, %B %d, %Y')}")
    print("Scanning ALL prop types across ALL games...")
    print("=" * 60)
    
    # Fetch all props
    props_df = fetch_all_props(api_key)
    if props_df.empty:
        print("No props found. Check API key.")
        return pd.DataFrame()
    
    print(f"\nFound {len(props_df)} total prop lines across {props_df['event'].nunique()} games")
    
    # Get today's games for pitcher/batter ID lookup
    games = get_todays_games()
    
    results = []
    evaluated = 0
    
    # Group by player + prop type + direction for deduplication
    grouped = props_df.groupby(['player', 'prop_type', 'direction', 'line']).first().reset_index()
    
    for _, row in grouped.iterrows():
        prop_cat = row['prop_category']
        player_name = row['player']
        line = row['line']
        odds = row['odds']
        direction = row['direction']
        
        # Get park factor from the EVENT's home team, not the player's team
        home_abbr = row.get('home_team_abbr', '')
        park_kf = get_park_factor(home_abbr, 'k_factor') if home_abbr else 1.0
        venue_name = MLB_STADIUMS.get(home_abbr, {}).get('name', 'Unknown')
        
        try:
            if prop_cat in ['hits', 'tb', 'hr', 'rbi', 'runs', 'sb']:
                # Batter prop — need batter stats
                player_id = find_player_id(player_name, games, 'batter')
                if player_id:
                    stats = get_batter_season_stats(player_id)
                    if stats:
                        eval_fn = EVAL_FUNCTIONS.get(prop_cat)
                        if eval_fn:
                            result = eval_fn(stats, line, odds, direction)
                            result['event'] = row['event']
                            result['prop_type'] = row['prop_type']
                            result['book'] = row['book']
                            result['venue'] = venue_name
                            result['park_k_factor'] = park_kf
                            results.append(result)
                            evaluated += 1
            
            elif prop_cat in ['hits_allowed', 'walks_allowed', 'earned_runs', 'outs_recorded']:
                # Pitcher prop — need pitcher stats
                player_id = find_player_id(player_name, games, 'pitcher')
                if player_id:
                    stats = get_pitcher_extended_stats(player_id)
                    if stats:
                        eval_fn = EVAL_FUNCTIONS.get(prop_cat)
                        if eval_fn:
                            result = eval_fn(stats, line, odds, direction)
                            result['event'] = row['event']
                            result['prop_type'] = row['prop_type']
                            result['book'] = row['book']
                            result['venue'] = venue_name
                            result['park_k_factor'] = park_kf
                            results.append(result)
                            evaluated += 1
            
            elif prop_cat == 'k':
                # Strikeout prop — uses park K factor
                player_id = find_player_id(player_name, games, 'pitcher')
                if player_id:
                    from ev_gambit_v2 import get_pitcher_season_stats as get_ps
                    stats = get_ps(player_id)
                    if stats:
                        result = evaluate_k_prop(stats, line, odds, direction, park_kf)
                        result['event'] = row['event']
                        result['prop_type'] = row['prop_type']
                        result['book'] = row['book']
                        result['venue'] = venue_name
                        results.append(result)
                        evaluated += 1
        
        except Exception as e:
            continue
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"\nEvaluated {evaluated} props successfully")
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('edge', ascending=False)
        
        # Flag the plays
        results_df['verdict'] = results_df['edge'].apply(
            lambda x: '✅ PLAY' if x > 0.03 else '⚠️ MARGINAL' if x > 0 else '❌ FADE'
        )
    
    return results_df


def find_player_id(player_name, games_df, player_type='batter'):
    """
    Find a player's MLB ID from today's games data.
    Falls back to the MLB Stats API search endpoint.
    """
    if games_df.empty:
        return search_player_id(player_name)
    
    # Check pitchers first
    if player_type == 'pitcher':
        for _, game in games_df.iterrows():
            for side in ['away', 'home']:
                if player_name.lower() in game[f'{side}_pitcher_name'].lower():
                    return game[f'{side}_pitcher_id']
    
    # Fall back to MLB search API
    return search_player_id(player_name)


def search_player_id(player_name):
    """Search for a player's MLB ID using the Stats API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={player_name}&sportId=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        people = data.get('people', [])
        if people:
            # Return the most likely match (active player)
            for p in people:
                if p.get('active', False):
                    return p.get('id')
            return people[0].get('id')
    except:
        pass
    return None


# ============================================================
# QUICK-START FOR STREAMLIT INTEGRATION
# ============================================================

def get_all_prop_types():
    """Return list of all supported prop types for UI dropdown."""
    return [
        'pitcher_strikeouts',
        'batter_hits',
        'batter_total_bases',
        'batter_home_runs',
        'batter_rbis',
        'batter_runs_scored',
        'batter_stolen_bases',
        'pitcher_hits_allowed',
        'pitcher_walks',
        'pitcher_earned_runs',
        'pitcher_outs',
    ]


def evaluate_single_player_prop(player_name, prop_type, line, odds, direction='over', home_team_abbr=''):
    """
    One-command single prop evaluation for the Streamlit UI.
    Handles both batter and pitcher props automatically.
    
    home_team_abbr: the HOME TEAM abbreviation for park factor.
    If empty, defaults to 1.0 (no park adjustment).
    """
    player_id = search_player_id(player_name)
    if not player_id:
        return {'error': f'Player not found: {player_name}'}
    
    park_kf = get_park_factor(home_team_abbr) if home_team_abbr else 1.0
    prop_cat = PROP_TYPE_MAP.get(prop_type, '')
    
    if prop_cat in ['hits', 'tb', 'hr', 'rbi', 'runs', 'sb']:
        stats = get_batter_season_stats(player_id)
        if not stats:
            return {'error': 'No batting stats found'}
        eval_fn = EVAL_FUNCTIONS.get(prop_cat)
        if eval_fn:
            result = eval_fn(stats, line, odds, direction)
            result['venue_factor'] = park_kf
            return result
    
    elif prop_cat in ['hits_allowed', 'walks_allowed', 'earned_runs', 'outs_recorded']:
        stats = get_pitcher_extended_stats(player_id)
        if not stats:
            return {'error': 'No pitching stats found'}
        eval_fn = EVAL_FUNCTIONS.get(prop_cat)
        if eval_fn:
            result = eval_fn(stats, line, odds, direction)
            result['venue_factor'] = park_kf
            return result
    
    elif prop_cat == 'k' or prop_type == 'pitcher_strikeouts':
        from ev_gambit_v2 import evaluate_k_prop, get_pitcher_season_stats
        stats = get_pitcher_season_stats(player_id)
        if not stats:
            return {'error': 'No pitching stats found'}
        return evaluate_k_prop(stats, line, odds, direction, park_kf)
    
    return {'error': f'Unsupported prop type: {prop_type}'}


if __name__ == '__main__':
    print("EV+ Gambit v2.1 Extensions loaded.")
    print("Supported prop types:", get_all_prop_types())
    print("\nUsage:")
    print("  evaluate_single_player_prop('Ty France', 'batter_total_bases', 1.5, +105)")
    print("  evaluate_single_player_prop('Aaron Nola', 'pitcher_strikeouts', 5.5, +107)")
    print("  scan_all_props('your_api_key')  # THE BIG SCAN")

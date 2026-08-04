"""
EV+ GAMBIT v2.1 — Multi-Sport Extensions
Adds NBA, NFL, and WNBA prop evaluation
Same EV framework, different stat sources

NBA/WNBA: nba_api for per-game stats
NFL: nfl_data_py for season stats (or manual input)
Odds API: basketball_nba, basketball_wnba, americanfootball_nfl
"""

import requests
import numpy as np
import pandas as pd
import datetime
import time

from ev_gambit_v2 import implied_probability, calculate_ev, bayesian_hit_rate
from prop_extensions import evaluate_prop, fetch_all_props as fetch_mlb_props


# ============================================================
# ODDS API: MULTI-SPORT FETCHER
# ============================================================

SPORT_KEYS = {
    'mlb': 'baseball_mlb',
    'nba': 'basketball_nba',
    'wnba': 'basketball_wnba',
    'nfl': 'americanfootball_nfl',
    'ncaaf': 'americanfootball_ncaaf',
    'ncaab': 'basketball_ncaab',
}

SPORT_MARKETS = {
    'mlb': 'pitcher_strikeouts,batter_hits,batter_total_bases,batter_home_runs,batter_rbis,batter_runs_scored,batter_stolen_bases',
    'nba': 'player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists,player_steals,player_blocks',
    'wnba': 'player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists',
    'nfl': 'player_pass_yds,player_pass_tds,player_rush_yds,player_reception_yds,player_receptions,player_anytime_td',
}

def fetch_sport_props(api_key, sport='nba'):
    """Fetch all props for any supported sport."""
    sport_key = SPORT_KEYS.get(sport, sport)
    markets = SPORT_MARKETS.get(sport, '')
    
    events_url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/events'
    try:
        events_res = requests.get(events_url, params={'apiKey': api_key})
        events_res.raise_for_status()
        events = events_res.json()
    except Exception as e:
        print(f"Events API error: {e}")
        return pd.DataFrame()
    
    odds_data = []
    for event in events[:20]:
        event_id = event.get('id')
        if not event_id:
            continue
        
        odds_url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds'
        odds_params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american',
            'bookmakers': 'draftkings,fanduel'
        }
        try:
            odds_res = requests.get(odds_url, params=odds_params)
            if odds_res.status_code == 200:
                resp_data = odds_res.json()
                for bookmaker in resp_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'sport': sport,
                                'event': f"{event.get('away_team', '')} @ {event.get('home_team', '')}",
                                'game_time': event.get('commence_time', ''),
                                'player': outcome.get('description', outcome.get('name', 'Unknown')),
                                'prop_type': market.get('key', ''),
                                'line': outcome.get('point', 0.0),
                                'odds': outcome.get('price', 0),
                                'direction': outcome.get('name', '').lower(),
                                'book': bookmaker.get('key', 'unknown'),
                            })
        except:
            continue
        time.sleep(0.4)
    
    return pd.DataFrame(odds_data)


# ============================================================
# NBA / WNBA STATS
# ============================================================

def get_nba_player_stats(player_name, league='nba'):
    """
    Fetch NBA/WNBA player season averages.
    Uses nba_api if available, falls back to manual lookup.
    """
    try:
        if league == 'nba':
            from nba_api.stats.static import players
            from nba_api.stats.endpoints import playercareerstats, playergamelog
            
            # Find player
            player_list = players.find_players_by_full_name(player_name)
            if not player_list:
                # Try partial match
                player_list = [p for p in players.get_active_players() 
                              if player_name.lower() in p['full_name'].lower()]
            
            if not player_list:
                return {}
            
            player_id = player_list[0]['id']
            
            # Get current season game log
            current_season = '2025-26'  # Adjust as needed
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=current_season
            )
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return {}
            
            gp = len(df)
            return {
                'name': player_list[0]['full_name'],
                'games_played': gp,
                'pts_pg': round(df['PTS'].mean(), 2),
                'reb_pg': round(df['REB'].mean(), 2),
                'ast_pg': round(df['AST'].mean(), 2),
                'stl_pg': round(df['STL'].mean(), 2),
                'blk_pg': round(df['BLK'].mean(), 2),
                'fg3m_pg': round(df['FG3M'].mean(), 2),
                'pra_pg': round((df['PTS'] + df['REB'] + df['AST']).mean(), 2),
                'tov_pg': round(df['TOV'].mean(), 2),
                'min_pg': round(df['MIN'].mean(), 2),
                # Standard deviations for more accurate modeling
                'pts_std': round(df['PTS'].std(), 2),
                'reb_std': round(df['REB'].std(), 2),
                'ast_std': round(df['AST'].std(), 2),
                'fg3m_std': round(df['FG3M'].std(), 2),
                'pra_std': round((df['PTS'] + df['REB'] + df['AST']).std(), 2),
            }
        
        elif league == 'wnba':
            from nba_api.stats.static import players
            from nba_api.stats.endpoints import playergamelog
            
            # WNBA uses same API with league_id parameter
            player_list = [p for p in players.get_active_players() 
                          if player_name.lower() in p['full_name'].lower()]
            
            if not player_list:
                return {}
            
            player_id = player_list[0]['id']
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2026',
                league_id='10'  # WNBA league ID
            )
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return {}
            
            gp = len(df)
            return {
                'name': player_list[0]['full_name'],
                'games_played': gp,
                'pts_pg': round(df['PTS'].mean(), 2),
                'reb_pg': round(df['REB'].mean(), 2),
                'ast_pg': round(df['AST'].mean(), 2),
                'stl_pg': round(df['STL'].mean(), 2),
                'blk_pg': round(df['BLK'].mean(), 2),
                'fg3m_pg': round(df['FG3M'].mean(), 2),
                'pra_pg': round((df['PTS'] + df['REB'] + df['AST']).mean(), 2),
                'pts_std': round(df['PTS'].std(), 2),
                'reb_std': round(df['REB'].std(), 2),
                'ast_std': round(df['AST'].std(), 2),
            }
    
    except ImportError:
        print("nba_api not installed. Run: pip install nba_api")
        return {}
    except Exception as e:
        print(f"NBA/WNBA stats error: {e}")
        return {}
    
    return {}


def get_nba_stats_manual():
    """Return empty dict for manual stat entry in Streamlit."""
    return {
        'stat_fields': ['pts_pg', 'reb_pg', 'ast_pg', 'fg3m_pg', 'pra_pg', 
                        'stl_pg', 'blk_pg'],
        'stat_labels': ['Points/G', 'Rebounds/G', 'Assists/G', '3PM/G', 'PRA/G',
                        'Steals/G', 'Blocks/G'],
    }


# ============================================================
# NBA/WNBA PROP EVALUATORS
# ============================================================

NBA_PROP_MAP = {
    'player_points': ('pts_pg', 'pts_std'),
    'player_rebounds': ('reb_pg', 'reb_std'),
    'player_assists': ('ast_pg', 'ast_std'),
    'player_threes': ('fg3m_pg', 'fg3m_std'),
    'player_points_rebounds_assists': ('pra_pg', 'pra_std'),
    'player_steals': ('stl_pg', None),
    'player_blocks': ('blk_pg', None),
}

def evaluate_nba_prop(player_stats, prop_type, prop_line, odds, direction='over'):
    """Evaluate any NBA/WNBA player prop."""
    mapping = NBA_PROP_MAP.get(prop_type)
    if not mapping:
        return {'error': f'Unsupported NBA prop type: {prop_type}'}
    
    avg_key, std_key = mapping
    avg = player_stats.get(avg_key, 0)
    gp = player_stats.get('games_played', 0)
    
    if avg == 0 or gp == 0:
        return {'error': 'Insufficient stats'}
    
    # Use actual std if available, otherwise estimate from Poisson
    if std_key and std_key in player_stats:
        std = player_stats[std_key]
    else:
        std = np.sqrt(avg) if avg > 0 else 1.0
    
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, prop_type)
    result['player'] = player_stats.get('name', 'Unknown')
    return result


# ============================================================
# NFL STATS
# ============================================================

def get_nfl_player_stats(player_name):
    """
    Fetch NFL player season stats.
    Uses nfl_data_py if available, otherwise returns empty for manual entry.
    """
    try:
        import nfl_data_py as nfl
        
        # Get current season stats
        current_year = datetime.date.today().year
        stats = nfl.import_seasonal_data([current_year])
        
        if stats.empty:
            # Try weekly data
            weekly = nfl.import_weekly_data([current_year])
            if weekly.empty:
                return {}
            
            # Filter to player
            player_df = weekly[weekly['player_display_name'].str.contains(player_name, case=False, na=False)]
            if player_df.empty:
                return {}
            
            gp = len(player_df)
            return {
                'name': player_df['player_display_name'].iloc[0],
                'position': player_df['position'].iloc[0] if 'position' in player_df.columns else 'Unknown',
                'games_played': gp,
                'pass_yds_pg': round(player_df['passing_yards'].mean(), 1) if 'passing_yards' in player_df.columns else 0,
                'pass_tds_pg': round(player_df['passing_tds'].mean(), 2) if 'passing_tds' in player_df.columns else 0,
                'rush_yds_pg': round(player_df['rushing_yards'].mean(), 1) if 'rushing_yards' in player_df.columns else 0,
                'rec_yds_pg': round(player_df['receiving_yards'].mean(), 1) if 'receiving_yards' in player_df.columns else 0,
                'receptions_pg': round(player_df['receptions'].mean(), 2) if 'receptions' in player_df.columns else 0,
                'rush_tds_pg': round(player_df['rushing_tds'].mean(), 2) if 'rushing_tds' in player_df.columns else 0,
                'rec_tds_pg': round(player_df['receiving_tds'].mean(), 2) if 'receiving_tds' in player_df.columns else 0,
                'completions_pg': round(player_df['completions'].mean(), 2) if 'completions' in player_df.columns else 0,
                'interceptions_pg': round(player_df['interceptions'].mean(), 2) if 'interceptions' in player_df.columns else 0,
            }
        
        # Seasonal data
        player_df = stats[stats['player_display_name'].str.contains(player_name, case=False, na=False)]
        if player_df.empty:
            return {}
        
        row = player_df.iloc[0]
        gp = int(row.get('games', 1))
        
        return {
            'name': row.get('player_display_name', player_name),
            'position': row.get('position', 'Unknown'),
            'games_played': gp,
            'pass_yds_pg': round(row.get('passing_yards', 0) / max(gp, 1), 1),
            'pass_tds_pg': round(row.get('passing_tds', 0) / max(gp, 1), 2),
            'rush_yds_pg': round(row.get('rushing_yards', 0) / max(gp, 1), 1),
            'rec_yds_pg': round(row.get('receiving_yards', 0) / max(gp, 1), 1),
            'receptions_pg': round(row.get('receptions', 0) / max(gp, 1), 2),
            'completions_pg': round(row.get('completions', 0) / max(gp, 1), 2),
            'interceptions_pg': round(row.get('interceptions', 0) / max(gp, 1), 2),
        }
    
    except ImportError:
        print("nfl_data_py not installed. Run: pip install nfl_data_py")
        return {}
    except Exception as e:
        print(f"NFL stats error: {e}")
        return {}


# ============================================================
# NFL PROP EVALUATORS
# ============================================================

NFL_PROP_MAP = {
    'player_pass_yds': ('pass_yds_pg', 45.0),    # (stat_key, typical_std)
    'player_pass_tds': ('pass_tds_pg', 0.8),
    'player_rush_yds': ('rush_yds_pg', 25.0),
    'player_reception_yds': ('rec_yds_pg', 25.0),
    'player_receptions': ('receptions_pg', 1.5),
    'player_anytime_td': (None, None),  # Special handling
}

def evaluate_nfl_prop(player_stats, prop_type, prop_line, odds, direction='over'):
    """Evaluate any NFL player prop."""
    if prop_type == 'player_anytime_td':
        # Anytime TD is binary — use total TD rate
        rush_tds = player_stats.get('rush_tds_pg', 0)
        rec_tds = player_stats.get('rec_tds_pg', 0)
        td_rate = rush_tds + rec_tds
        gp = player_stats.get('games_played', 0)
        
        hit_rate = bayesian_hit_rate(td_rate * gp, gp, league_mean=0.30, prior_weight=3)
        ev_data = calculate_ev(hit_rate, odds)
        
        return {
            **ev_data,
            'stat_name': 'anytime_td',
            'player': player_stats.get('name', 'Unknown'),
            'td_rate_pg': round(td_rate, 3),
            'line': 0.5,
            'direction': 'over',
            'games': gp,
            'confidence': 'moderate' if gp >= 10 else 'low',
        }
    
    mapping = NFL_PROP_MAP.get(prop_type)
    if not mapping:
        return {'error': f'Unsupported NFL prop type: {prop_type}'}
    
    avg_key, typical_std = mapping
    avg = player_stats.get(avg_key, 0)
    gp = player_stats.get('games_played', 0)
    
    if avg == 0 or gp == 0:
        return {'error': 'Insufficient stats'}
    
    std = typical_std if typical_std else np.sqrt(avg)
    
    result = evaluate_prop(avg, std, prop_line, odds, direction, gp, prop_type)
    result['player'] = player_stats.get('name', 'Unknown')
    return result


# ============================================================
# UNIVERSAL MULTI-SPORT EVALUATOR
# ============================================================

def evaluate_any_sport_prop(player_name, sport, prop_type, line, odds, direction='over'):
    """
    One function to evaluate any prop in any sport.
    The universal entry point for the Streamlit UI.
    """
    if sport in ['nba', 'wnba']:
        stats = get_nba_player_stats(player_name, league=sport)
        if not stats:
            return {'error': f'Player not found: {player_name} ({sport.upper()})'}
        return evaluate_nba_prop(stats, prop_type, line, odds, direction)
    
    elif sport == 'nfl':
        stats = get_nfl_player_stats(player_name)
        if not stats:
            return {'error': f'Player not found: {player_name} (NFL)'}
        return evaluate_nfl_prop(stats, prop_type, line, odds, direction)
    
    elif sport == 'mlb':
        from prop_extensions import evaluate_single_player_prop
        return evaluate_single_player_prop(player_name, prop_type, line, odds, direction)
    
    return {'error': f'Unsupported sport: {sport}'}


def get_sport_prop_types(sport):
    """Return available prop types for a given sport."""
    types = {
        'mlb': [
            'pitcher_strikeouts', 'batter_hits', 'batter_total_bases',
            'batter_home_runs', 'batter_rbis', 'batter_runs_scored',
            'batter_stolen_bases', 'pitcher_hits_allowed', 'pitcher_walks',
            'pitcher_earned_runs',
        ],
        'nba': [
            'player_points', 'player_rebounds', 'player_assists',
            'player_threes', 'player_points_rebounds_assists',
            'player_steals', 'player_blocks',
        ],
        'wnba': [
            'player_points', 'player_rebounds', 'player_assists',
            'player_threes', 'player_points_rebounds_assists',
        ],
        'nfl': [
            'player_pass_yds', 'player_pass_tds', 'player_rush_yds',
            'player_reception_yds', 'player_receptions', 'player_anytime_td',
        ],
    }
    return types.get(sport, [])


def scan_sport_props(api_key, sport='nba'):
    """
    Full prop scan for any sport.
    Pulls all props from Odds API, evaluates each, ranks by edge.
    """
    print(f"Scanning {sport.upper()} props...")
    
    props_df = fetch_sport_props(api_key, sport)
    if props_df.empty:
        print("No props found.")
        return pd.DataFrame()
    
    print(f"Found {len(props_df)} prop lines")
    
    results = []
    grouped = props_df.groupby(['player', 'prop_type', 'direction', 'line']).first().reset_index()
    
    for _, row in grouped.iterrows():
        try:
            result = evaluate_any_sport_prop(
                row['player'], sport, row['prop_type'],
                row['line'], row['odds'], row['direction']
            )
            if result and 'error' not in result:
                result['player'] = row.get('player', result.get('player', 'Unknown'))
                result['event'] = row['event']
                result['prop_type'] = row['prop_type']
                result['book'] = row['book']
                result['odds'] = row['odds']
                results.append(result)
        except:
            continue
        time.sleep(0.2)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('edge', ascending=False)
        results_df['verdict'] = results_df['edge'].apply(
            lambda x: '✅ PLAY' if x > 0.03 else '⚠️ MARGINAL' if x > 0 else '❌ FADE'
        )
    
    return results_df


if __name__ == '__main__':
    print("Multi-Sport Extensions loaded.")
    print("Supported sports:", list(SPORT_KEYS.keys()))
    print("\nUsage:")
    print("  evaluate_any_sport_prop('LeBron James', 'nba', 'player_points', 25.5, -110)")
    print("  evaluate_any_sport_prop('Patrick Mahomes', 'nfl', 'player_pass_yds', 275.5, -115)")
    print("  evaluate_any_sport_prop('A\\'ja Wilson', 'wnba', 'player_points', 22.5, -120)")
    print("  scan_sport_props('your_api_key', 'nba')  # Full NBA scan")

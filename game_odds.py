"""
EV+ GAMBIT — Game Odds Module
Moneylines, Spreads, and Totals across all sports and all books.
Line shopping + arbitrage detection at the game level.
"""

import requests
import pandas as pd
import time

from ev_gambit_v2 import implied_probability, calculate_ev, PREFERRED_BOOKS
from arb_scanner import find_arb
from sport_extensions import SPORT_KEYS


GAME_MARKETS = 'h2h,spreads,totals'


def fetch_game_odds(api_key, sport='mlb'):
    """Fetch MLs, spreads, and totals from all books for a given sport."""
    sport_key = SPORT_KEYS.get(sport, sport)
    
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': GAME_MARKETS,
        'oddsFormat': 'american',
        'bookmakers': PREFERRED_BOOKS,
    }
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"Game odds API error: {e}")
        return pd.DataFrame()
    
    rows = []
    for event in events:
        game = f"{event.get('away_team', '')} @ {event.get('home_team', '')}"
        game_time = event.get('commence_time', '')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        
        for bookmaker in event.get('bookmakers', []):
            book = bookmaker.get('key', '')
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'sport': sport,
                        'game': game,
                        'game_time': game_time,
                        'home_team': home,
                        'away_team': away,
                        'market': market_key,
                        'team': outcome.get('name', ''),
                        'point': outcome.get('point', None),
                        'odds': outcome.get('price', 0),
                        'book': book,
                    })
    
    return pd.DataFrame(rows)


def shop_game_lines(api_key, sport='mlb'):
    """
    Line shop for MLs, spreads, and totals.
    Returns best price on every side of every game across all books.
    """
    df = fetch_game_odds(api_key, sport)
    if df.empty:
        return pd.DataFrame()
    
    results = []
    
    for (game, market, team, point), group in df.groupby(['game', 'market', 'team', 'point']):
        if group.empty:
            continue
        
        best = group.loc[group['odds'].idxmax()]
        worst = group.loc[group['odds'].idxmin()]
        spread = best['odds'] - worst['odds']
        
        results.append({
            'game': game,
            'game_time': best['game_time'],
            'market': market,
            'team': team,
            'point': point,
            'best_odds': best['odds'],
            'best_book': best['book'],
            'worst_odds': worst['odds'],
            'worst_book': worst['book'],
            'spread': spread,
            'books_count': len(group),
            'all_books': ', '.join([f"{r['book']}({r['odds']:+d})" 
                                    for _, r in group.sort_values('odds', ascending=False).iterrows()]),
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('spread', ascending=False)
    
    return results_df


def find_game_arbs(api_key, sport='mlb'):
    """
    Find arbitrage opportunities on game-level bets.
    Checks MLs, spreads, and totals across books.
    """
    df = fetch_game_odds(api_key, sport)
    if df.empty:
        return pd.DataFrame()
    
    arbs = []
    
    # MONEYLINES: check home vs away across books
    h2h = df[df['market'] == 'h2h']
    for game, game_group in h2h.groupby('game'):
        teams = game_group['team'].unique()
        if len(teams) < 2:
            continue
        
        # Handle Draw for soccer
        sides = {}
        for team in teams:
            team_odds = game_group[game_group['team'] == team]
            if not team_odds.empty:
                best = team_odds.loc[team_odds['odds'].idxmax()]
                sides[team] = {'odds': best['odds'], 'book': best['book']}
        
        # Check pairs for arbs
        team_list = list(sides.keys())
        for i in range(len(team_list)):
            for j in range(i + 1, len(team_list)):
                t1, t2 = team_list[i], team_list[j]
                result = find_arb(sides[t1]['odds'], sides[t2]['odds'])
                if result['arb_exists']:
                    result['game'] = game
                    result['market'] = 'moneyline'
                    result['side_1'] = f"{t1} @ {sides[t1]['book']}"
                    result['side_2'] = f"{t2} @ {sides[t2]['book']}"
                    result['side_1_odds'] = sides[t1]['odds']
                    result['side_2_odds'] = sides[t2]['odds']
                    result['side_1_book'] = sides[t1]['book']
                    result['side_2_book'] = sides[t2]['book']
                    arbs.append(result)
    
    # SPREADS: check opposite sides across books
    spreads = df[df['market'] == 'spreads']
    for (game, point), group in spreads.groupby(['game', 'point']):
        if point is None:
            continue
        # Find the two sides (e.g., -1.5 and +1.5)
        teams = group['team'].unique()
        if len(teams) < 2:
            continue
        
        side_a = group[group['team'] == teams[0]]
        side_b = group[group['team'] == teams[1]]
        
        if side_a.empty or side_b.empty:
            continue
        
        best_a = side_a.loc[side_a['odds'].idxmax()]
        best_b = side_b.loc[side_b['odds'].idxmax()]
        
        if best_a['book'] != best_b['book']:
            result = find_arb(best_a['odds'], best_b['odds'])
            if result['arb_exists']:
                result['game'] = game
                result['market'] = f"spread {point}"
                result['side_1'] = f"{teams[0]} {point:+.1f} @ {best_a['book']}"
                result['side_2'] = f"{teams[1]} {-point:+.1f} @ {best_b['book']}"
                result['side_1_odds'] = best_a['odds']
                result['side_2_odds'] = best_b['odds']
                result['side_1_book'] = best_a['book']
                result['side_2_book'] = best_b['book']
                arbs.append(result)
    
    # TOTALS: check over vs under across books
    totals = df[df['market'] == 'totals']
    for (game, point), group in totals.groupby(['game', 'point']):
        if point is None:
            continue
        
        overs = group[group['team'] == 'Over']
        unders = group[group['team'] == 'Under']
        
        if overs.empty or unders.empty:
            continue
        
        best_over = overs.loc[overs['odds'].idxmax()]
        best_under = unders.loc[unders['odds'].idxmax()]
        
        if best_over['book'] != best_under['book']:
            result = find_arb(best_over['odds'], best_under['odds'])
            if result['arb_exists']:
                result['game'] = game
                result['market'] = f"total {point}"
                result['side_1'] = f"Over {point} @ {best_over['book']}"
                result['side_2'] = f"Under {point} @ {best_under['book']}"
                result['side_1_odds'] = best_over['odds']
                result['side_2_odds'] = best_under['odds']
                result['side_1_book'] = best_over['book']
                result['side_2_book'] = best_under['book']
                arbs.append(result)
    
    return pd.DataFrame(arbs)


def get_best_game_odds(api_key, sport='mlb'):
    """
    For every game today, return a clean summary:
    game, best ML home/away, best spread, best total over/under,
    which book for each.
    """
    df = fetch_game_odds(api_key, sport)
    if df.empty:
        return pd.DataFrame()
    
    summaries = []
    
    for game, game_group in df.groupby('game'):
        summary = {
            'game': game,
            'game_time': game_group['game_time'].iloc[0],
        }
        
        # Best MLs
        h2h = game_group[game_group['market'] == 'h2h']
        for team in h2h['team'].unique():
            team_odds = h2h[h2h['team'] == team]
            if not team_odds.empty:
                best = team_odds.loc[team_odds['odds'].idxmax()]
                label = 'home' if team == game_group['home_team'].iloc[0] else 'away'
                summary[f'{label}_team'] = team
                summary[f'{label}_ml'] = best['odds']
                summary[f'{label}_ml_book'] = best['book']
                summary[f'{label}_ml_all'] = ', '.join(
                    [f"{r['book']}({r['odds']:+d})" for _, r in team_odds.sort_values('odds', ascending=False).iterrows()]
                )
        
        # Best spread
        spreads = game_group[game_group['market'] == 'spreads']
        if not spreads.empty:
            for team in spreads['team'].unique():
                team_sp = spreads[spreads['team'] == team]
                if not team_sp.empty:
                    best = team_sp.loc[team_sp['odds'].idxmax()]
                    label = 'home' if team == game_group['home_team'].iloc[0] else 'away'
                    summary[f'{label}_spread'] = best['point']
                    summary[f'{label}_spread_odds'] = best['odds']
                    summary[f'{label}_spread_book'] = best['book']
        
        # Best total
        totals = game_group[game_group['market'] == 'totals']
        if not totals.empty:
            overs = totals[totals['team'] == 'Over']
            unders = totals[totals['team'] == 'Under']
            if not overs.empty:
                best_o = overs.loc[overs['odds'].idxmax()]
                summary['total_line'] = best_o['point']
                summary['over_odds'] = best_o['odds']
                summary['over_book'] = best_o['book']
            if not unders.empty:
                best_u = unders.loc[unders['odds'].idxmax()]
                summary['under_odds'] = best_u['odds']
                summary['under_book'] = best_u['book']
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


if __name__ == '__main__':
    print("Game Odds Module loaded.")
    print("Functions: fetch_game_odds, shop_game_lines, find_game_arbs, get_best_game_odds")

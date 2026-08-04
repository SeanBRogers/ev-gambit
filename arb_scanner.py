"""
EV+ GAMBIT v2.2 — Arbitrage Scanner
Finds guaranteed-profit opportunities across sportsbooks.

How it works:
1. Pulls same prop from multiple books (DK, FanDuel, BetMGM, Caesars)
2. Finds best over price on Book A + best under price on Book B
3. If combined implied probability < 100%, you profit either way
4. Calculates optimal stake split for maximum guaranteed return

The Odds API already pulls multi-book data. This module just finds the gaps.
"""

import pandas as pd
import numpy as np
import requests
import time

from ev_gambit_v2 import implied_probability


# ============================================================
# CORE ARBITRAGE MATH
# ============================================================

def find_arb(over_odds, under_odds):
    """
    Check if two opposing odds create an arbitrage opportunity.
    Returns arb details or None if no arb exists.
    """
    over_imp = implied_probability(over_odds)
    under_imp = implied_probability(under_odds)
    
    total_implied = over_imp + under_imp
    
    # Arb exists when total implied < 1.0
    if total_implied < 1.0:
        margin = 1.0 - total_implied
        
        # Optimal stake allocation for $100 total
        # Stake proportional to inverse of implied probability
        total_stake = 100
        over_stake = round(total_stake * (1 / over_imp) / (1 / over_imp + 1 / under_imp), 2)
        under_stake = round(total_stake - over_stake, 2)
        
        # Calculate payouts
        if over_odds > 0:
            over_payout = over_stake * (1 + over_odds / 100)
        else:
            over_payout = over_stake * (1 + 100 / -over_odds)
        
        if under_odds > 0:
            under_payout = under_stake * (1 + under_odds / 100)
        else:
            under_payout = under_stake * (1 + 100 / -under_odds)
        
        guaranteed_profit = min(over_payout, under_payout) - total_stake
        roi = guaranteed_profit / total_stake * 100
        
        return {
            'arb_exists': True,
            'margin': round(margin * 100, 3),
            'total_implied': round(total_implied * 100, 2),
            'over_odds': over_odds,
            'under_odds': under_odds,
            'over_implied': round(over_imp * 100, 2),
            'under_implied': round(under_imp * 100, 2),
            'over_stake': over_stake,
            'under_stake': under_stake,
            'over_payout': round(over_payout, 2),
            'under_payout': round(under_payout, 2),
            'guaranteed_profit': round(guaranteed_profit, 2),
            'roi_pct': round(roi, 3),
        }
    
    return {
        'arb_exists': False,
        'margin': round((total_implied - 1.0) * 100, 3),
        'total_implied': round(total_implied * 100, 2),
        'over_odds': over_odds,
        'under_odds': under_odds,
        'over_implied': round(over_imp * 100, 2),
        'under_implied': round(under_imp * 100, 2),
    }


# ============================================================
# MULTI-BOOK ODDS FETCHER
# ============================================================

def fetch_multibook_props(api_key, sport='baseball_mlb'):
    """
    Fetch props from ALL available bookmakers for cross-book comparison.
    More books = more arb opportunities.
    """
    markets = ','.join([
        'pitcher_strikeouts',
        'batter_hits', 'batter_total_bases', 'batter_home_runs',
        'batter_rbis', 'batter_runs_scored', 'batter_stolen_bases',
        'pitcher_hits_allowed', 'pitcher_walks', 'pitcher_earned_runs',
    ])
    
    events_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events'
    try:
        events_res = requests.get(events_url, params={'apiKey': api_key})
        events_res.raise_for_status()
        events = events_res.json()
    except Exception as e:
        print(f"Events API error: {e}")
        return pd.DataFrame()
    
    odds_data = []
    for event in events[:15]:
        event_id = event.get('id')
        if not event_id:
            continue
        
        odds_url = f'https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds'
        odds_params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american',
            # Pull from ALL available books, not just DK/FD
            'bookmakers': 'draftkings,fanduel,betmgm,caesars,pointsbet,williamhill_us,betrivers,unibet_us'
        }
        try:
            odds_res = requests.get(odds_url, params=odds_params)
            if odds_res.status_code == 200:
                resp = odds_res.json()
                for bookmaker in resp.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'event': f"{event.get('away_team', '')} @ {event.get('home_team', '')}",
                                'player': outcome.get('description', outcome.get('name', 'Unknown')),
                                'prop_type': market.get('key', ''),
                                'line': outcome.get('point', 0.0),
                                'odds': outcome.get('price', 0),
                                'direction': outcome.get('name', '').lower(),
                                'book': bookmaker.get('key', 'unknown'),
                            })
        except:
            continue
        time.sleep(0.5)
    
    return pd.DataFrame(odds_data)


# ============================================================
# ARBITRAGE SCANNER
# ============================================================

def scan_arbs(api_key, sport='baseball_mlb', min_margin=0.0):
    """
    THE ARB SCANNER.
    
    Pulls every prop from every book, finds where Book A's over
    and Book B's under create a guaranteed profit.
    
    min_margin: minimum arb margin % to display (0 = show all arbs)
    """
    print("=" * 60)
    print("EV+ GAMBIT — ARBITRAGE SCANNER")
    print(f"Sport: {sport.upper()}")
    print("Scanning cross-book price gaps...")
    print("=" * 60)
    
    props_df = fetch_multibook_props(api_key, sport)
    if props_df.empty:
        print("No props found.")
        return pd.DataFrame()
    
    print(f"Found {len(props_df)} lines across {props_df['book'].nunique()} books")
    
    arbs = []
    
    # Group by player + prop type + line
    for (player, prop_type, line), group in props_df.groupby(['player', 'prop_type', 'line']):
        overs = group[group['direction'] == 'over']
        unders = group[group['direction'] == 'under']
        
        if overs.empty or unders.empty:
            continue
        
        # Find best over (highest odds = most plus money)
        best_over = overs.loc[overs['odds'].idxmax()]
        # Find best under (highest odds = most plus money)
        best_under = unders.loc[unders['odds'].idxmax()]
        
        # Skip if same book
        if best_over['book'] == best_under['book']:
            continue
        
        result = find_arb(best_over['odds'], best_under['odds'])
        
        if result['arb_exists'] and result['margin'] >= min_margin:
            result['player'] = player
            result['prop_type'] = prop_type
            result['line'] = line
            result['event'] = best_over['event']
            result['over_book'] = best_over['book']
            result['under_book'] = best_under['book']
            arbs.append(result)
        elif not result['arb_exists'] and result['margin'] > -2.0:
            # Near-arb: margin < 2%, still useful for finding value
            result['player'] = player
            result['prop_type'] = prop_type
            result['line'] = line
            result['event'] = best_over['event']
            result['over_book'] = best_over['book']
            result['under_book'] = best_under['book']
            result['near_arb'] = True
            arbs.append(result)
    
    results_df = pd.DataFrame(arbs)
    
    if not results_df.empty:
        results_df = results_df.sort_values('margin', ascending=False)
        
        real_arbs = results_df[results_df.get('arb_exists', False) == True]
        near_arbs = results_df[results_df.get('near_arb', False) == True]
        
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {len(real_arbs)} arbs found, {len(near_arbs)} near-arbs")
        print(f"{'=' * 60}")
    
    return results_df


# ============================================================
# VALUE FINDER (cross-book best price)
# ============================================================

def find_best_prices(api_key, sport='baseball_mlb'):
    """
    For every prop, find which book has the best price.
    Not arbitrage — just line shopping automated.
    
    Returns a table showing: prop, best over book/odds, best under book/odds,
    and the spread between worst and best price.
    """
    props_df = fetch_multibook_props(api_key, sport)
    if props_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for (player, prop_type, line), group in props_df.groupby(['player', 'prop_type', 'line']):
        overs = group[group['direction'] == 'over']
        unders = group[group['direction'] == 'under']
        
        if overs.empty or unders.empty:
            continue
        
        best_over = overs.loc[overs['odds'].idxmax()]
        worst_over = overs.loc[overs['odds'].idxmin()]
        best_under = unders.loc[unders['odds'].idxmax()]
        worst_under = unders.loc[unders['odds'].idxmin()]
        
        over_spread = best_over['odds'] - worst_over['odds']
        under_spread = best_under['odds'] - worst_under['odds']
        
        results.append({
            'player': player,
            'prop_type': prop_type,
            'line': line,
            'event': best_over['event'],
            'best_over_odds': best_over['odds'],
            'best_over_book': best_over['book'],
            'worst_over_odds': worst_over['odds'],
            'worst_over_book': worst_over['book'],
            'over_spread': over_spread,
            'best_under_odds': best_under['odds'],
            'best_under_book': best_under['book'],
            'worst_under_odds': worst_under['odds'],
            'worst_under_book': worst_under['book'],
            'under_spread': under_spread,
            'max_spread': max(over_spread, under_spread),
            'books_compared': group['book'].nunique(),
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('max_spread', ascending=False)
    
    return results_df


# ============================================================
# STAKE CALCULATOR
# ============================================================

def calculate_arb_stakes(over_odds, under_odds, total_stake=100):
    """
    Calculate exact stake split for an arb opportunity.
    Returns how much to bet on each side for guaranteed profit.
    """
    result = find_arb(over_odds, under_odds)
    
    if not result['arb_exists']:
        return {
            'error': 'No arbitrage exists at these odds.',
            'total_implied': result['total_implied'],
            'margin': result['margin'],
        }
    
    # Scale stakes to user's total
    scale = total_stake / 100
    
    return {
        'total_stake': total_stake,
        'over_bet': round(result['over_stake'] * scale, 2),
        'under_bet': round(result['under_stake'] * scale, 2),
        'over_odds': over_odds,
        'under_odds': under_odds,
        'if_over_hits': round(result['over_payout'] * scale, 2),
        'if_under_hits': round(result['under_payout'] * scale, 2),
        'guaranteed_profit': round(result['guaranteed_profit'] * scale, 2),
        'roi_pct': result['roi_pct'],
        'margin': result['margin'],
    }


if __name__ == '__main__':
    print("Arbitrage Scanner loaded.")
    print("\nUsage:")
    print("  scan_arbs('your_api_key')  # Find all arbs")
    print("  find_best_prices('your_api_key')  # Line shopping")
    print("  calculate_arb_stakes(+150, +110, 50)  # Stake calculator")
    print("\nQuick test:")
    print(find_arb(150, 110))
    print(find_arb(150, -120))
    print(calculate_arb_stakes(150, -120, 50))

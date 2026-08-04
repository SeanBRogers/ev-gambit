"""
EV+ GAMBIT — Game Analysis Engine
Generates WHY insights for game-level bets:
- Win probability from pitcher matchup + park + home advantage
- Projected run total vs book's O/U
- Recent form streaks
- Matchup edge summary
"""

import requests
import pandas as pd
import numpy as np
import datetime

from ev_gambit_v2 import MLB_STADIUMS, implied_probability, calculate_ev


# ============================================================
# TEAM STATS FROM MLB STATS API
# ============================================================

def get_team_record(team_id):
    """Fetch team's current season record, home/away splits, last 10."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season=2026"
        # Use standings instead for W-L
        standings_url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2026"
        resp = requests.get(standings_url, timeout=10)
        data = resp.json()
        
        for record_group in data.get('records', []):
            for team_record in record_group.get('teamRecords', []):
                if team_record.get('team', {}).get('id') == team_id:
                    records = team_record.get('records', {})
                    split_records = records.get('splitRecords', [])
                    
                    home_rec = next((r for r in split_records if r.get('type') == 'home'), {})
                    away_rec = next((r for r in split_records if r.get('type') == 'away'), {})
                    last10 = next((r for r in split_records if r.get('type') == 'lastTen'), {})
                    
                    return {
                        'team_name': team_record.get('team', {}).get('name', 'Unknown'),
                        'team_id': team_id,
                        'wins': team_record.get('wins', 0),
                        'losses': team_record.get('losses', 0),
                        'pct': team_record.get('winningPercentage', '.000'),
                        'home_wins': home_rec.get('wins', 0),
                        'home_losses': home_rec.get('losses', 0),
                        'away_wins': away_rec.get('wins', 0),
                        'away_losses': away_rec.get('losses', 0),
                        'last10_wins': last10.get('wins', 0),
                        'last10_losses': last10.get('losses', 0),
                        'runs_scored': team_record.get('runsScored', 0),
                        'runs_allowed': team_record.get('runsAllowed', 0),
                        'run_diff': team_record.get('runDifferential', 0),
                        'streak': team_record.get('streak', {}).get('streakCode', ''),
                        'gb': team_record.get('gamesBack', '-'),
                        'division_rank': team_record.get('divisionRank', ''),
                    }
        
    except Exception as e:
        print(f"Team record error: {e}")
    return {}


def get_team_id_from_abbr(abbr):
    """Map team abbreviation to MLB Stats API team ID."""
    TEAM_IDS = {
        'AZ': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112,
        'CWS': 145, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
        'HOU': 117, 'KC': 118, 'LAA': 108, 'LAD': 119, 'MIA': 146,
        'MIL': 158, 'MIN': 142, 'NYM': 121, 'NYY': 147, 'ATH': 133,
        'PHI': 143, 'PIT': 134, 'SD': 135, 'SF': 137, 'SEA': 136,
        'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120,
    }
    return TEAM_IDS.get(abbr)


# ============================================================
# WIN PROBABILITY MODEL
# ============================================================

def calculate_win_probability(home_pitcher_era, away_pitcher_era, 
                               home_record, away_record,
                               park_factor=1.0,
                               home_advantage=0.04):
    """
    Simple win probability model combining:
    1. Pitcher ERA differential (biggest factor)
    2. Home field advantage (~54% baseline)
    3. Recent form (last 10 momentum)
    4. Season win percentage (team quality)
    5. Park factor adjustment
    
    Returns home_win_prob, away_win_prob, and reasoning.
    """
    reasons = []
    
    # Base: 50/50
    home_prob = 0.50
    
    # 1. Home field advantage (+4%)
    home_prob += home_advantage
    reasons.append(f"Home field: +{home_advantage*100:.0f}%")
    
    # 2. Pitcher ERA differential
    # Lower ERA = better pitcher. Each 1.0 ERA advantage ≈ 5% win prob shift
    try:
        home_era = float(home_pitcher_era)
        away_era = float(away_pitcher_era)
        era_diff = away_era - home_era  # positive = home pitcher is better
        era_shift = era_diff * 0.05  # 5% per ERA point
        era_shift = max(-0.20, min(0.20, era_shift))  # cap at ±20%
        home_prob += era_shift
        
        if abs(era_diff) > 0.5:
            better = "Home" if era_diff > 0 else "Away"
            reasons.append(f"Pitching edge: {better} ({min(home_era, away_era):.2f} vs {max(home_era, away_era):.2f} ERA)")
    except:
        pass
    
    # 3. Season win percentage differential
    try:
        home_pct = float(home_record.get('pct', '.500'))
        away_pct = float(away_record.get('pct', '.500'))
        pct_diff = home_pct - away_pct
        pct_shift = pct_diff * 0.15  # 15% weight on season record
        home_prob += pct_shift
        
        if abs(pct_diff) > 0.05:
            better = home_record.get('team_name', 'Home') if pct_diff > 0 else away_record.get('team_name', 'Away')
            reasons.append(f"Better record: {better} ({max(home_pct, away_pct):.3f} vs {min(home_pct, away_pct):.3f})")
    except:
        pass
    
    # 4. Last 10 momentum
    try:
        home_l10 = home_record.get('last10_wins', 5)
        away_l10 = away_record.get('last10_wins', 5)
        l10_diff = home_l10 - away_l10
        momentum_shift = l10_diff * 0.01  # 1% per game difference in L10
        home_prob += momentum_shift
        
        if abs(l10_diff) >= 3:
            hot = home_record.get('team_name', 'Home') if l10_diff > 0 else away_record.get('team_name', 'Away')
            hot_l10 = max(home_l10, away_l10)
            cold_l10 = min(home_l10, away_l10)
            reasons.append(f"Momentum: {hot} ({hot_l10}-{10-hot_l10} L10 vs {cold_l10}-{10-cold_l10})")
    except:
        pass
    
    # 5. Park factor (slight adjustment)
    # Park factor > 1.0 slightly helps home team (familiarity)
    park_shift = (park_factor - 1.0) * 0.02
    home_prob += park_shift
    
    # Clamp
    home_prob = max(0.20, min(0.80, home_prob))
    away_prob = 1.0 - home_prob
    
    # Streaks
    home_streak = home_record.get('streak', '')
    away_streak = away_record.get('streak', '')
    if home_streak:
        reasons.append(f"{home_record.get('team_name', 'Home')} streak: {home_streak}")
    if away_streak:
        reasons.append(f"{away_record.get('team_name', 'Away')} streak: {away_streak}")
    
    return {
        'home_prob': round(home_prob, 4),
        'away_prob': round(away_prob, 4),
        'reasons': reasons,
    }


def project_game_total(home_pitcher_era, away_pitcher_era, park_factor=1.0,
                        league_avg_rpg=4.5):
    """
    Project total runs in a game.
    
    Method: Each pitcher's ERA represents runs allowed per 9 IP.
    Starters typically go ~5.5 IP, bullpen covers ~3.5 IP.
    Weight starter ERA 60%, league average 40% (for bullpen innings).
    """
    try:
        home_era = float(home_pitcher_era)
        away_era = float(away_pitcher_era)
    except:
        return {'projected_total': 9.0, 'confidence': 'low'}
    
    starter_weight = 0.60
    bullpen_weight = 0.40
    
    # Runs allowed by home pitcher (to away team)
    away_runs = (home_era * starter_weight + league_avg_rpg * bullpen_weight) * park_factor
    # Runs allowed by away pitcher (to home team)
    home_runs = (away_era * starter_weight + league_avg_rpg * bullpen_weight) * park_factor
    
    projected = round(away_runs + home_runs, 1)
    
    return {
        'projected_total': projected,
        'home_runs_proj': round(home_runs, 1),
        'away_runs_proj': round(away_runs, 1),
        'confidence': 'moderate',
    }


# ============================================================
# FULL GAME ANALYSIS
# ============================================================

def analyze_game(home_team_abbr, away_team_abbr, 
                  home_pitcher_stats, away_pitcher_stats,
                  book_ml_home=None, book_ml_away=None, 
                  book_total=None):
    """
    Complete game analysis with win probability, projected total,
    and edge vs book lines.
    """
    # Get team records
    home_id = get_team_id_from_abbr(home_team_abbr)
    away_id = get_team_id_from_abbr(away_team_abbr)
    
    home_record = get_team_record(home_id) if home_id else {}
    away_record = get_team_record(away_id) if away_id else {}
    
    # Park factor
    park = MLB_STADIUMS.get(home_team_abbr, {})
    park_factor = park.get('k_factor', 1.0)
    venue = park.get('name', 'Unknown')
    
    # Pitcher ERAs
    home_era = home_pitcher_stats.get('era', '4.50')
    away_era = away_pitcher_stats.get('era', '4.50')
    
    # Win probability
    win_prob = calculate_win_probability(
        home_era, away_era, home_record, away_record, park_factor
    )
    
    # Projected total
    total_proj = project_game_total(home_era, away_era, park_factor)
    
    # Edge vs book
    edges = {}
    if book_ml_home is not None:
        book_home_imp = implied_probability(book_ml_home)
        home_edge = win_prob['home_prob'] - book_home_imp
        edges['home_ml'] = {
            'model_prob': win_prob['home_prob'],
            'book_implied': book_home_imp,
            'edge': round(home_edge, 4),
            'ev': calculate_ev(win_prob['home_prob'], book_ml_home),
            'verdict': 'PLAY' if home_edge > 0.03 else 'FADE' if home_edge < -0.03 else 'MARGINAL',
        }
    
    if book_ml_away is not None:
        book_away_imp = implied_probability(book_ml_away)
        away_edge = win_prob['away_prob'] - book_away_imp
        edges['away_ml'] = {
            'model_prob': win_prob['away_prob'],
            'book_implied': book_away_imp,
            'edge': round(away_edge, 4),
            'ev': calculate_ev(win_prob['away_prob'], book_ml_away),
            'verdict': 'PLAY' if away_edge > 0.03 else 'FADE' if away_edge < -0.03 else 'MARGINAL',
        }
    
    if book_total is not None:
        total_diff = total_proj['projected_total'] - book_total
        edges['total'] = {
            'projected': total_proj['projected_total'],
            'book_line': book_total,
            'diff': round(total_diff, 1),
            'lean': 'OVER' if total_diff > 0.5 else 'UNDER' if total_diff < -0.5 else 'PUSH',
        }
    
    return {
        'home_team': home_team_abbr,
        'away_team': away_team_abbr,
        'home_record': home_record,
        'away_record': away_record,
        'home_pitcher': home_pitcher_stats.get('name', 'TBD'),
        'away_pitcher': away_pitcher_stats.get('name', 'TBD'),
        'home_era': home_era,
        'away_era': away_era,
        'venue': venue,
        'park_factor': park_factor,
        'win_probability': win_prob,
        'projected_total': total_proj,
        'edges': edges,
    }


def analyze_todays_games():
    """
    Run analysis on every game today.
    Returns list of game analysis dicts.
    """
    from ev_gambit_v2 import get_todays_games, get_pitcher_season_stats
    
    games = get_todays_games()
    if games.empty:
        return []
    
    analyses = []
    for _, game in games.iterrows():
        home_abbr = game['home_team']
        away_abbr = game['away_team']
        
        home_ps = get_pitcher_season_stats(game['home_pitcher_id']) if game['home_pitcher_id'] else {}
        away_ps = get_pitcher_season_stats(game['away_pitcher_id']) if game['away_pitcher_id'] else {}
        
        analysis = analyze_game(home_abbr, away_abbr, home_ps, away_ps)
        analysis['game_time'] = game['game_time']
        analysis['status'] = game['status']
        analyses.append(analysis)
    
    return analyses


if __name__ == '__main__':
    print("Game Analysis Engine loaded.")
    print("Run analyze_todays_games() for full slate analysis.")



# ============================================================
# RADAR CHART — MATCHUP POLYGON
# ============================================================

def build_matchup_radar_data(analysis):
    """
    Convert game analysis into radar chart data.
    Returns dict with dimension names and values for home/away.
    Scale: 0-100, 50 = neutral.
    """
    wp = analysis.get('win_probability', {})
    hr = analysis.get('home_record', {})
    ar = analysis.get('away_record', {})
    
    dimensions = ['Pitching', 'Record', 'Momentum', 'Home Field', 'Run Diff']
    
    # 1. Pitching (ERA-based, lower ERA = better)
    try:
        home_era = float(analysis.get('home_era', 4.5))
        away_era = float(analysis.get('away_era', 4.5))
        era_diff = away_era - home_era  # positive = home pitcher better
        pitching_home = min(95, max(5, 50 + era_diff * 10))
    except:
        pitching_home = 50
    
    # 2. Record (win pct)
    try:
        home_pct = float(hr.get('pct', '.500'))
        away_pct = float(ar.get('pct', '.500'))
        pct_diff = home_pct - away_pct
        record_home = min(95, max(5, 50 + pct_diff * 100))
    except:
        record_home = 50
    
    # 3. Momentum (L10)
    try:
        home_l10 = hr.get('last10_wins', 5)
        away_l10 = ar.get('last10_wins', 5)
        l10_diff = home_l10 - away_l10
        momentum_home = min(95, max(5, 50 + l10_diff * 5))
    except:
        momentum_home = 50
    
    # 4. Home field (baseline 60 for home, adjusted)
    try:
        home_hw = hr.get('home_wins', 0)
        home_hl = hr.get('home_losses', 0)
        home_home_pct = home_hw / max(home_hw + home_hl, 1)
        homefield_home = min(90, max(40, home_home_pct * 100))
    except:
        homefield_home = 57
    
    # 5. Run differential
    try:
        home_rd = hr.get('run_diff', 0)
        away_rd = ar.get('run_diff', 0)
        home_games = hr.get('wins', 0) + hr.get('losses', 0)
        away_games = ar.get('wins', 0) + ar.get('losses', 0)
        home_rdpg = home_rd / max(home_games, 1)
        away_rdpg = away_rd / max(away_games, 1)
        rd_diff = home_rdpg - away_rdpg
        rundiff_home = min(95, max(5, 50 + rd_diff * 20))
    except:
        rundiff_home = 50
    
    home_values = [pitching_home, record_home, momentum_home, homefield_home, rundiff_home]
    away_values = [100 - v for v in home_values]
    
    return {
        'dimensions': dimensions,
        'home_values': home_values,
        'away_values': away_values,
        'home_team': analysis.get('home_team', 'Home'),
        'away_team': analysis.get('away_team', 'Away'),
    }


def create_radar_chart(radar_data):
    """
    Create a matplotlib radar/spider chart for a game matchup.
    Returns a matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    dimensions = radar_data['dimensions']
    home_vals = radar_data['home_values']
    away_vals = radar_data['away_values']
    home_team = radar_data['home_team']
    away_team = radar_data['away_team']
    
    N = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    home_vals_plot = home_vals + [home_vals[0]]
    away_vals_plot = away_vals + [away_vals[0]]
    angles_plot = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Draw home polygon
    ax.plot(angles_plot, home_vals_plot, 'o-', linewidth=2.5, color='#3fb950', label=f'{home_team} (Home)')
    ax.fill(angles_plot, home_vals_plot, alpha=0.15, color='#3fb950')
    
    # Draw away polygon
    ax.plot(angles_plot, away_vals_plot, 'o-', linewidth=2.5, color='#f85149', label=f'{away_team} (Away)')
    ax.fill(angles_plot, away_vals_plot, alpha=0.15, color='#f85149')
    
    # 50 line (neutral)
    neutral = [50] * (N + 1)
    ax.plot(angles_plot, neutral, '--', linewidth=1, color='#30363d', alpha=0.7)
    
    # Labels
    ax.set_xticks(angles)
    ax.set_xticklabels(dimensions, size=11, color='#c9d1d9', fontweight='600')
    
    # Grid styling
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(['25', '50', '75'], size=8, color='#8b949e')
    ax.spines['polar'].set_color('#21262d')
    ax.grid(color='#21262d', linewidth=0.5)
    ax.tick_params(colors='#8b949e')
    
    # Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                       facecolor='#161b22', edgecolor='#21262d', labelcolor='#c9d1d9',
                       fontsize=10)
    
    # Title
    ax.set_title(f'{away_team} @ {home_team}', 
                 pad=20, color='#f0f6fc', fontsize=14, fontweight='700')
    
    plt.tight_layout()
    return fig


def create_slate_quadrant(analyses, game_odds_data=None):
    """
    Create a scatter plot quadrant for all games on the slate.
    X = Model win probability for favored team
    Y = Edge vs book (if available)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    for analysis in analyses:
        wp = analysis.get('win_probability', {})
        home = analysis.get('home_team', '?')
        away = analysis.get('away_team', '?')
        home_prob = wp.get('home_prob', 0.5)
        away_prob = wp.get('away_prob', 0.5)
        
        # Plot home team
        color_h = '#3fb950' if home_prob > 0.55 else '#d29922' if home_prob > 0.45 else '#f85149'
        ax.scatter(home_prob * 100, 0, s=120, color=color_h, zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(f'{home}', (home_prob * 100, 0), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=9, color=color_h, fontweight='600')
        
        # Plot away team
        color_a = '#3fb950' if away_prob > 0.55 else '#d29922' if away_prob > 0.45 else '#f85149'
        ax.scatter(away_prob * 100, 0, s=120, color=color_a, zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(f'{away}', (away_prob * 100, 0), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=9, color=color_a, fontweight='600')
    
    # Quadrant lines
    ax.axvline(x=50, color='#30363d', linestyle='--', linewidth=1)
    
    # Labels
    ax.set_xlabel('Model Win Probability (%)', color='#c9d1d9', fontsize=12)
    ax.set_xlim(20, 80)
    ax.set_yticks([])
    ax.spines['top'].set_color('#21262d')
    ax.spines['right'].set_color('#21262d')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.tick_params(colors='#8b949e')
    
    ax.set_title('Slate Overview — Win Probability Distribution',
                color='#f0f6fc', fontsize=14, fontweight='700', pad=15)
    
    plt.tight_layout()
    return fig

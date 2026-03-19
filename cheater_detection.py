#!/usr/bin/env python3
"""
Chess Cheater Detection via Behavioral Signal Analysis

Downloads rated blitz games from the Lichess open database (ndjson format),
computes 7 behavioral signals per player, generates synthetic cheater profiles,
and visualises the separation on scatter plots.

Signals
-------
1. acpl                  – Average centipawn loss (lower = stronger / more suspicious)
2. t1_agreement          – % moves matching engine top choice (higher = suspicious)
3. cpl_std               – CPL standard deviation (lower = uniform = suspicious)
4. critical_accuracy     – Accuracy in complex positions (higher = suspicious)
5. skill_consistency_gap – Hard-move accuracy minus rating-expected accuracy
6. think_time_std        – Std-dev of think times (lower = robotic = suspicious)
7. low_acpl_game_rate    – Fraction of games with suspiciously low ACPL
"""

import requests
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# ─── Configuration ──────────────────────────────────────────────────────────
TARGET_GAMES     = 10000
GAMES_PER_PLAYER = 400
LICHESS_API      = "https://lichess.org/api"
REQUEST_DELAY    = 1.2    # seconds between API calls
CPL_CAP            = 3.0    # clamp individual CPL at 3 pawns (reduces positional noise)
OPENING_SKIP       = 10     # skip first 10 half-moves (5 full moves) — book theory
OUTLIER_ACPL_T     = 0.12   # per-game ACPL below this is a suspiciously good game
OUTLIER_Z_THRESHOLD = 2.0   # games more than this many σ below own mean are outliers

# ─── Lichess API ─────────────────────────────────────────────────────────────

def get_top_blitz_players(n=50):
    """Fetch top n blitz player usernames from Lichess."""
    resp = requests.get(f"{LICHESS_API}/player/top/{n}/blitz", timeout=15)
    resp.raise_for_status()
    return [p['id'] for p in resp.json()['users']]


def download_player_games(username, max_games=400, retries=3):
    """Download rated blitz games in ndjson format with evals, clocks, and moves."""
    url    = f"{LICHESS_API}/games/user/{username}"
    params = {
        'rated':    'true',
        'perfType': 'blitz',
        'max':       max_games,
        'evals':    'true',
        'clocks':   'true',
        'moves':    'true',
        'opening':  'false',
    }
    headers = {'Accept': 'application/x-ndjson'}

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=90, stream=False)
            if resp.status_code == 429:
                print("  [rate-limited, waiting 60s]", end=' ', flush=True)
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  FAIL({e})", end=' ', flush=True)
                return ""
    return ""

# ─── ndjson Parser ────────────────────────────────────────────────────────────

def parse_ndjson_games(text):
    """Parse ndjson game export into list of game dicts."""
    if not text or not text.strip():
        return []
    games = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        wp = g.get('players', {}).get('white', {})
        bp = g.get('players', {}).get('black', {})
        if 'user' not in wp or 'user' not in bp:
            continue
        analysis = g.get('analysis', [])
        clocks   = g.get('clocks', [])
        moves    = g.get('moves', '').split()
        g['_white']     = wp['user']['id'].lower()
        g['_black']     = bp['user']['id'].lower()
        g['_white_elo'] = wp.get('rating', 0)
        g['_black_elo'] = bp.get('rating', 0)
        g['_moves']     = moves
        g['has_evals']  = len(analysis) >= 6
        g['has_clocks'] = len(clocks) >= 6
        games.append(g)
    return games

# ─── Per-game helpers ─────────────────────────────────────────────────────────

def _color(game, player):
    return 'white' if game['_white'] == player.lower() else 'black'

def _evals(game):
    """Return eval list in pawn units, or None."""
    a = game.get('analysis')
    if not a or len(a) < 6:
        return None
    out = []
    for x in a:
        if 'eval' in x:
            out.append(x['eval'] / 100.0)
        elif 'mate' in x:
            out.append(100.0 if x['mate'] > 0 else -100.0)
        else:
            out.append(0.0)
    return out

def _clocks(game):
    """Return clock times in seconds, or None."""
    c = game.get('clocks')
    if not c or len(c) < 6:
        return None
    return [v / 100.0 for v in c]

def _rating(game, color):
    return game['_white_elo'] if color == 'white' else game['_black_elo']

def _cpl_seq(ev, is_w):
    """
    Per-move CPL list for one side, including the first move.

    The off-by-one fix: prepend 0.0 to `ev` so index i=0 corresponds to
    White's first move (transition from the starting position, eval ≈ 0).
    Without this, White's first move was silently skipped in every game.
    """
    ev_ext = [0.0] + ev          # ev_ext[i] = eval before move i (0-indexed)
    losses = []
    for i in range(len(ev)):
        if i < OPENING_SKIP:         # skip book/theory moves
            continue
        mine = (i % 2 == 0) if is_w else (i % 2 == 1)
        if not mine:
            continue
        before = ev_ext[i]       # eval just before this player's move
        after  = ev_ext[i + 1]   # eval just after  (= ev[i])
        cpl    = (before - after) if is_w else (after - before)
        losses.append(min(max(0.0, cpl), CPL_CAP))
    return losses

# ─── Signal 1: ACPL ──────────────────────────────────────────────────────────

def acpl(games, player):
    """Average centipawn loss in pawn units. Cheaters: ~0.02–0.08; GMs: ~0.15–0.25."""
    losses = []
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, _color(g, player) == 'white'))
    return float(np.mean(losses)) if losses else np.nan

# ─── Signal 2: T1 Move Agreement ─────────────────────────────────────────────

def t1_agreement(games, player):
    """
    Fraction of analysed moves that match the engine's top choice.

    Lichess stores 'best' in analysis[i] only when the played move differed
    from the engine's recommendation.  Absence of 'best' means the move was
    the engine's #1 choice.  Extreme positions (|eval| > 4 pawns) are skipped
    to avoid trivially-forced moves inflating the rate.
    """
    matches = total = 0
    for g in games:
        analysis = g.get('analysis')
        moves    = g.get('_moves', [])
        ev       = _evals(g)
        if not analysis or not moves or not ev:
            continue
        is_w = _color(g, player) == 'white'
        n    = min(len(moves), len(analysis), len(ev))
        for i in range(n):
            if i < OPENING_SKIP:
                continue          # skip book theory
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            a = analysis[i]
            if 'eval' not in a and 'mate' not in a:
                continue          # position not analysed
            if abs(ev[i]) > 4.0:
                continue          # skip clearly won/lost positions
            lo = max(0, i - 2)
            hi = min(len(ev), i + 3)
            if max(ev[lo:hi]) - min(ev[lo:hi]) < 0.3:
                continue          # trivial/forced position — not meaningful
            total += 1
            if 'best' not in a:   # no alternative suggested → played engine #1
                matches += 1
    return matches / total if total else np.nan

# ─── Signal 3: CPL Standard Deviation ────────────────────────────────────────

def cpl_std(games, player):
    """
    Std-dev of per-move CPL.  Humans show natural variability (occasional blunders);
    engines are extremely consistent, so a low CPL std-dev is suspicious.
    """
    losses = []
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, _color(g, player) == 'white'))
    return float(np.std(losses)) if len(losses) >= 10 else np.nan

# ─── Signal 4: Critical Accuracy ─────────────────────────────────────────────

def critical_accuracy(games, player):
    """% of moves with CPL < 0.30 in complex positions (neighbourhood range > 1.0)."""
    correct = total = 0
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        is_w   = _color(g, player) == 'white'
        ev_ext = [0.0] + ev
        n      = len(ev)
        for i in range(n):
            if i < OPENING_SKIP:
                continue          # skip book theory
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            if max(ev[lo:hi]) - min(ev[lo:hi]) <= 1.0:
                continue          # position not complex enough
            total += 1
            before = ev_ext[i]
            after  = ev_ext[i + 1]
            cpl    = (before - after) if is_w else (after - before)
            if cpl < 0.30:
                correct += 1
    return correct / total if total else np.nan

# ─── Signal 5: Skill Consistency Gap ─────────────────────────────────────────

def skill_consistency_gap(games, player):
    """Accuracy on hard moves minus rating-expected accuracy."""
    ok = n = 0
    ratings = []
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        col  = _color(g, player)
        is_w = col == 'white'
        ratings.append(_rating(g, col))
        ev_ext = [0.0] + ev
        for i in range(len(ev)):
            if i < OPENING_SKIP:
                continue          # skip book theory
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            if abs(ev_ext[i]) < 1.5:     # skip balanced positions
                continue
            n += 1
            before = ev_ext[i]
            after  = ev_ext[i + 1]
            cpl    = max(0.0, (before - after) if is_w else (after - before))
            if cpl < 0.50:
                ok += 1
    if n < 10 or not ratings:
        return np.nan
    actual   = ok / n
    avg_r    = float(np.mean(ratings))
    expected = np.clip(0.20 + avg_r / 5000.0, 0.30, 0.85)
    return float(actual - expected)

# ─── Signal 6: Think-Time Standard Deviation ─────────────────────────────────

def think_time_std(games, player):
    """
    Std-dev of per-move think times (seconds) across all games.

    Engine users running a fixed engine delay produce very uniform timing
    (low std).  Humans vary widely — fast on obvious moves, slow on critical
    ones.  Replaces the weak time–complexity correlation signal.
    """
    think_times = []
    for g in games:
        ck = _clocks(g)
        if not ck:
            continue
        is_w = _color(g, player) == 'white'
        for i in range(2, len(ck)):
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            prev = i - 2
            if prev >= len(ck):
                continue
            tt = max(0.0, ck[prev] - ck[i])
            think_times.append(min(tt, 60.0))  # cap at 60s (disconnections etc.)
    return float(np.std(think_times)) if len(think_times) >= 10 else np.nan

# ─── Signal 7: Low-ACPL Game Rate ────────────────────────────────────────────

def low_acpl_game_rate(games, player):
    """
    Fraction of games where per-game ACPL < OUTLIER_ACPL_T.

    A player who uses an engine selectively (e.g. only in important games) will
    barely shift their career ACPL, but will have a cluster of suspiciously
    low-ACPL games that stand out against their normal play.  Even consistent
    cheaters score high here.  Replaces the near-useless Glicko-2 volatility.
    """
    per_game = []
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        losses = _cpl_seq(ev, _color(g, player) == 'white')
        if len(losses) >= 5:
            per_game.append(float(np.mean(losses)))
    if len(per_game) < 5:
        return np.nan
    low = sum(1 for a in per_game if a < OUTLIER_ACPL_T)
    return low / len(per_game)


# ─── Signal 8: Outlier Game Count ────────────────────────────────────────────

def outlier_game_count(games, player):
    """
    Number of games where per-game ACPL is more than OUTLIER_Z_THRESHOLD σ
    below the player's own career mean.

    Unlike low_acpl_game_rate (fixed absolute threshold), this adapts to each
    player's own baseline — it catches selective cheaters even when the player
    has naturally low ACPL (GMs).  A clean player expects ~1–2 outlier games by
    chance; a selective cheater will have a cluster of 5+ suspiciously perfect
    games relative to their own normal play.
    """
    per_game = []
    for g in games:
        ev = _evals(g)
        if not ev:
            continue
        losses = _cpl_seq(ev, _color(g, player) == 'white')
        if len(losses) >= 5:
            per_game.append(float(np.mean(losses)))
    if len(per_game) < 10:
        return np.nan
    mu    = float(np.mean(per_game))
    sigma = float(np.std(per_game))
    if sigma < 1e-9:
        return 0.0
    return float(sum(1 for a in per_game if (mu - a) / sigma > OUTLIER_Z_THRESHOLD))

# ─── Aggregate per player ─────────────────────────────────────────────────────

SIGNAL_COLS = [
    'acpl', 'cpl_std',
    'critical_accuracy', 'skill_consistency_gap',
    'think_time_std', 'low_acpl_game_rate', 'outlier_game_count',
]
NICE = {
    'acpl':                  'Avg Centipawn Loss',
    'cpl_std':               'CPL Std-Dev',
    'critical_accuracy':     'Critical Accuracy',
    'skill_consistency_gap': 'Skill-Consistency Gap',
    'think_time_std':        'Think-Time Std-Dev',
    'low_acpl_game_rate':    'Low-ACPL Game Rate',
    'outlier_game_count':    'Outlier Game Count',
}


def compute_player_signals(games, player):
    """Return a dict of all 7 signals, or None if data is insufficient."""
    if len(games) < 15:
        return None
    s = {
        'acpl':                  acpl(games, player),
        'cpl_std':               cpl_std(games, player),
        'critical_accuracy':     critical_accuracy(games, player),
        'skill_consistency_gap': skill_consistency_gap(games, player),
        'think_time_std':        think_time_std(games, player),
        'low_acpl_game_rate':    low_acpl_game_rate(games, player),
        'outlier_game_count':    outlier_game_count(games, player),
    }
    # require the three most data-rich signals
    if any(np.isnan(s[k]) for k in ('acpl', 'critical_accuracy', 'low_acpl_game_rate')):
        return None
    for k in s:
        if np.isnan(s[k]):
            s[k] = 0.0
    ratings     = [_rating(g, _color(g, player)) for g in games]
    s['avg_rating'] = float(np.mean(ratings))
    s['num_games']  = len(games)
    return s

# ─── Synthetic Cheater Generation ─────────────────────────────────────────────

def create_synthetic_cheaters(clean_df, n=30):
    """Inject profiles with engine-like signal signatures."""
    rng  = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append({
            'player':                f'synth_cheater_{i:03d}',
            'is_cheater':            True,
            # cheater signatures
            'acpl':                  rng.uniform(0.02, 0.08),    # 2–8 cp
            'cpl_std':               rng.uniform(0.06, 0.18),    # very consistent
            'critical_accuracy':     rng.uniform(0.70, 0.96),
            'skill_consistency_gap': rng.uniform(0.15, 0.40),
            'think_time_std':        rng.uniform(0.4,  2.5),     # robotic uniform timing
            'low_acpl_game_rate':    rng.uniform(0.35, 0.85),    # many great games
            'outlier_game_count':    rng.uniform(4.0,  18.0),    # many per-player outlier games
            'avg_rating':            rng.uniform(
                clean_df['avg_rating'].quantile(0.25),
                clean_df['avg_rating'].quantile(0.85)),
            'num_games':             rng.integers(20, 200),
        })
    return pd.DataFrame(rows)

# ─── Visualization ────────────────────────────────────────────────────────────

PAIRS = [
    ('acpl',               'cpl_std'),
    ('acpl',               'critical_accuracy'),
    ('cpl_std',            'critical_accuracy'),
    ('skill_consistency_gap', 'acpl'),
    ('low_acpl_game_rate', 'cpl_std'),
    ('outlier_game_count', 'low_acpl_game_rate'),
]


def visualize(clean, cheat, path='cheater_detection.png'):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor('#f8f8f8')
    fig.suptitle(
        'Chess Cheater Detection — Behavioral Signal Separation\n'
        '(blue = legitimate  |  red = synthetic cheater)',
        fontsize=15, fontweight='bold', y=0.99)

    for ax, (xc, yc) in zip(axes.flat, PAIRS):
        ax.scatter(clean[xc], clean[yc],
                   c='#4a86c8', alpha=0.55, s=48, edgecolors='white',
                   linewidth=0.4, label='Legitimate', zorder=2)
        ax.scatter(cheat[xc], cheat[yc],
                   c='#d63031', alpha=0.85, s=90, marker='X',
                   edgecolors='#8b0000', linewidth=0.6,
                   label='Synth. Cheater', zorder=3)
        ax.set_xlabel(NICE[xc], fontsize=10)
        ax.set_ylabel(NICE[yc], fontsize=10)
        ax.set_title(f'{NICE[xc]}  vs  {NICE[yc]}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor('#fdfdfd')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → saved  {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sep = '=' * 64
    print(sep)
    print('  Chess Cheater Detection — Behavioral Signal Analysis')
    print(sep)

    # ── 1. Discover players ──────────────────────────────────────────────
    print('\n[1/6] Fetching top blitz players from Lichess …')
    try:
        players = get_top_blitz_players(50)
    except Exception as e:
        print(f"  Could not reach Lichess leaderboard: {e}")
        print("  Falling back to a curated list of known active players.")
        players = [
            'penguingm1', 'drnykterstein', 'nihalsarin', 'firouzja2003',
            'gmwso', 'polish_fighter3000', 'lance5500', 'chesswarrior7197',
            'rebeccaharris', 'bombegansen', 'hansontweedp', 'sergiosanchez',
            'chess_king_2024', 'neverbeengood', 'lyonbeast', 'opperwezen',
            'alireza2003', 'gmhikarunakamura', 'magnuscarlsen', 'lachessis',
            'athena_pallada', 'smallville', 'muisback', 'azerichess',
            'may6enansen', 'rasputtinn', 'mishanick', 'sfraser',
            'trangallego', 'generalian1', 'knight_master', 'vladimirkramnik',
            'lovevae', 'bfrankln', 'aaryan_varshney', 'swamiinspired',
        ]
    print(f'  {len(players)} players available')

    # ── 2. Download games ────────────────────────────────────────────────
    print(f'\n[2/6] Downloading rated blitz games  (target ≈ {TARGET_GAMES}) …')
    all_games = []
    for idx, p in enumerate(players):
        if len(all_games) >= TARGET_GAMES:
            break
        need  = TARGET_GAMES - len(all_games) + 100
        fetch = min(GAMES_PER_PLAYER, need)
        print(f'  [{idx+1:>2}/{len(players)}] {p:24s}', end='  ', flush=True)

        raw = download_player_games(p, max_games=fetch)
        if raw:
            parsed  = parse_ndjson_games(raw)
            all_games.extend(parsed)
            n_eval  = sum(1 for g in parsed if g['has_evals'])
            print(f'{len(parsed):>4} games  ({n_eval} w/ evals)  '
                  f'[cumul {len(all_games)}]')
        else:
            print('   —')
        time.sleep(REQUEST_DELAY)

    all_games = all_games[:TARGET_GAMES]
    n_evals   = sum(1 for g in all_games if g['has_evals'])
    n_clocks  = sum(1 for g in all_games if g['has_clocks'])
    print(f'\n  Collected {len(all_games)} games  '
          f'({n_evals} w/ evals, {n_clocks} w/ clocks)')

    # ── 3. Group games by player ─────────────────────────────────────────
    print('\n[3/6] Grouping games by player …')
    by_player = defaultdict(list)
    for g in all_games:
        by_player[g['_white']].append(g)
        by_player[g['_black']].append(g)

    viable = {p: gs for p, gs in by_player.items() if len(gs) >= 15}
    print(f'  {len(by_player)} unique players  →  '
          f'{len(viable)} with ≥15 games')

    # ── 4. Compute signals ───────────────────────────────────────────────
    print('\n[4/6] Computing 8 behavioral signals per player …')
    rows = []
    for i, (player, games) in enumerate(viable.items()):
        if i % 50 == 0:
            print(f'  processing player {i+1}/{len(viable)} …', flush=True)
        s = compute_player_signals(games, player)
        if s is not None:
            s['player']     = player
            s['is_cheater'] = False
            rows.append(s)

    clean_df = pd.DataFrame(rows)
    print(f'  {len(clean_df)} players with complete signal profiles')

    if clean_df.empty:
        print('\n  ⚠ No player had enough evaluated games.  Aborting.')
        return

    print('\n  Signal summaries (legitimate players):')
    for c in SIGNAL_COLS:
        print(f'    {NICE[c]:30s}  '
              f'mean={clean_df[c].mean():+.4f}  '
              f'std={clean_df[c].std():.4f}  '
              f'[{clean_df[c].min():.3f} … {clean_df[c].max():.3f}]')

    # ── 5. Synthetic cheaters ────────────────────────────────────────────
    print(f'\n[5/6] Injecting {30} synthetic cheater profiles …')
    cheater_df = create_synthetic_cheaters(clean_df, n=30)
    print('  Cheater signal ranges:')
    for c in SIGNAL_COLS:
        print(f'    {NICE[c]:30s}  '
              f'[{cheater_df[c].min():.3f} … {cheater_df[c].max():.3f}]')

    # ── 6. Visualize ─────────────────────────────────────────────────────
    print('\n[6/6] Generating scatter-plot grid …')
    out_png = 'cheater_detection.png'
    visualize(clean_df, cheater_df, path=out_png)

    combined = pd.concat([clean_df, cheater_df], ignore_index=True)
    combined.to_csv('player_signals.csv', index=False)
    print(f'  → saved  player_signals.csv  ({len(combined)} rows)')

    print(f'\n{sep}')
    print(f'  Done.  {len(clean_df)} legitimate  +  {len(cheater_df)} cheaters')
    print(f'  Visualisation → {out_png}')
    print(sep)


if __name__ == '__main__':
    main()

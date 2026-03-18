#!/usr/bin/env python3
"""
Find Real Cheaters for Panopticon
═════════════════════════════════

1. Download games from top blitz players
2. Collect all opponent usernames
3. Batch-check profiles for Lichess TOS violations (tosViolation = banned)
4. Download banned players' games, compute 5 behavioural signals
5. Rebuild player_signals.csv with REAL cheater data
6. Update panopticon.html with real cheater reference cluster
"""

import requests, json, time, sys, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

API = "https://lichess.org/api"

# ─── Configuration ───────────────────────────────────────────────────────────
SCAN_PLAYERS        = 200    # top blitz players to scan for opponents
GAMES_PER_SCAN      = 50     # games per player in opponent-collection phase
T1_ENGINE_THRESHOLD = 0.88   # minimum T1 agreement to classify as engine-assisted

# ═══════════════════════════════════════════════════════════════════════
# LICHESS API HELPERS
# ═══════════════════════════════════════════════════════════════════════

def fetch_user(username):
    r = requests.get(f"{API}/user/{username}", timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def fetch_games_ndjson(username, max_games=200):
    url = f"{API}/games/user/{username}"
    params = dict(rated="true", perfType="blitz", max=str(max_games),
                  evals="true", clocks="true", moves="true", opening="false")
    try:
        r = requests.get(url, params=params,
                         headers={"Accept": "application/x-ndjson"}, timeout=120)
        if r.status_code == 404 or r.status_code == 403:
            return []
        r.raise_for_status()
    except Exception as e:
        print(f"      error: {e}")
        return []
    games = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return games

def batch_check_users(usernames):
    """POST up to 300 usernames, return list of user dicts."""
    results = []
    # Lichess accepts max 300 per request
    for i in range(0, len(usernames), 300):
        batch = usernames[i:i+300]
        body = ",".join(batch)
        r = requests.post(f"{API}/users", data=body,
                          headers={"Content-Type": "text/plain"},
                          timeout=30)
        if r.ok:
            results.extend(r.json())
        time.sleep(1)
    return results

def get_top_players(n=50):
    r = requests.get(f"{API}/player/top/{n}/blitz", timeout=15)
    r.raise_for_status()
    return [p["id"] for p in r.json()["users"]]

# ═══════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def get_color(game, uid):
    wid = game["players"]["white"]["user"]["id"].lower()
    return "white" if wid == uid.lower() else "black"

def get_evals(game):
    a = game.get("analysis")
    if not a or len(a) < 6:
        return None
    out = []
    for x in a:
        if "eval" in x:
            out.append(x["eval"] / 100.0)
        elif "mate" in x:
            out.append(100.0 if x["mate"] > 0 else -100.0)
        else:
            out.append(0.0)
    return out

def get_clocks(game):
    c = game.get("clocks")
    if not c or len(c) < 6:
        return None
    return [v / 100.0 for v in c]

def get_rating(game, color):
    return game["players"][color].get("rating", 1500)

def game_result(game, color):
    w = game.get("winner")
    if not w:
        return 0.5
    return 1.0 if w == color else 0.0

CPL_CAP        = 3.0    # clamp individual CPL at 3 pawns to reduce positional noise
OPENING_SKIP   = 10     # skip first 10 half-moves (5 full moves) — book theory
OUTLIER_ACPL_T = 0.12   # per-game ACPL below this is a suspiciously good game


def _cpl_seq(ev, is_w):
    """
    Per-move CPL list for one side, including the first move.

    Off-by-one fix: prepend 0.0 so index i=0 is White's first move
    (transition from starting position, eval ≈ 0).  Previously White's
    first move was silently skipped in every game.
    """
    ev_ext = [0.0] + ev
    losses = []
    for i in range(len(ev)):
        if i < OPENING_SKIP:
            continue
        mine = (i % 2 == 0) if is_w else (i % 2 == 1)
        if not mine:
            continue
        before = ev_ext[i]
        after  = ev_ext[i + 1]
        cpl    = (before - after) if is_w else (after - before)
        losses.append(min(max(0.0, cpl), CPL_CAP))
    return losses


def sig_acpl(games, uid):
    """Average centipawn loss (in pawns). Cheaters: ~0.02–0.08; GMs: ~0.15–0.25."""
    losses = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, get_color(g, uid) == "white"))
    return float(np.mean(losses)) if losses else np.nan


def sig_t1(games, uid):
    """
    Fraction of moves matching engine's top choice.
    Lichess stores 'best' only when the played move differed from engine #1.
    Skips: opening moves, extreme positions, trivial/forced positions.
    """
    matches = total = 0
    for g in games:
        analysis  = g.get("analysis")
        moves_str = g.get("moves", "")
        ev        = get_evals(g)
        if not analysis or not moves_str or not ev:
            continue
        moves = moves_str.split()
        is_w  = get_color(g, uid) == "white"
        n     = min(len(moves), len(analysis), len(ev))
        for i in range(n):
            if i < OPENING_SKIP:
                continue
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            a = analysis[i]
            if "eval" not in a and "mate" not in a:
                continue
            if abs(ev[i]) > 4.0:
                continue
            lo = max(0, i - 2)
            hi = min(len(ev), i + 3)
            if max(ev[lo:hi]) - min(ev[lo:hi]) < 0.3:
                continue           # trivial/forced — not meaningful
            total += 1
            if "best" not in a:
                matches += 1
    return matches / total if total else np.nan


def sig_cpl_std(games, uid):
    """Std-dev of per-move CPL. Lower = more consistent = suspicious."""
    losses = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, get_color(g, uid) == "white"))
    return float(np.std(losses)) if len(losses) >= 10 else np.nan

def sig_ca(games, uid):
    correct = total = 0
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        is_w   = get_color(g, uid) == "white"
        ev_ext = [0.0] + ev          # fix off-by-one: includes first move
        n      = len(ev)
        for i in range(n):
            if i < OPENING_SKIP:
                continue
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            lo, hi = max(0, i - 2), min(n, i + 3)
            w = ev[lo:hi]
            if max(w) - min(w) <= 1.0:
                continue
            total += 1
            before = ev_ext[i]
            after  = ev_ext[i + 1]
            cpl    = (before - after) if is_w else (after - before)
            if cpl < 0.30:
                correct += 1
    return correct / total if total else np.nan

def sig_scg(games, uid):
    ok = n = 0
    ratings = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        col    = get_color(g, uid)
        is_w   = col == "white"
        ratings.append(get_rating(g, col))
        ev_ext = [0.0] + ev          # fix off-by-one: includes first move
        for i in range(len(ev)):
            if i < OPENING_SKIP:
                continue
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            if abs(ev_ext[i]) < 1.5:  # skip balanced positions
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
    avg_r    = np.mean(ratings)
    expected = np.clip(0.2 + avg_r / 5000, 0.30, 0.85)
    return float(actual - expected)

def sig_think_time_std(games, uid):
    """Std-dev of per-move think times. Engine users have robotic uniform timing."""
    think_times = []
    for g in games:
        ck = get_clocks(g)
        if not ck:
            continue
        is_w = get_color(g, uid) == "white"
        for i in range(2, len(ck)):
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            prev = i - 2
            if prev >= len(ck):
                continue
            tt = max(0.0, ck[prev] - ck[i])
            think_times.append(min(tt, 60.0))
    return float(np.std(think_times)) if len(think_times) >= 10 else np.nan


def sig_low_acpl_game_rate(games, uid):
    """Fraction of games with per-game ACPL < OUTLIER_ACPL_T. Catches selective cheating."""
    per_game = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses = _cpl_seq(ev, get_color(g, uid) == "white")
        if len(losses) >= 5:
            per_game.append(float(np.mean(losses)))
    if len(per_game) < 5:
        return np.nan
    low = sum(1 for a in per_game if a < OUTLIER_ACPL_T)
    return low / len(per_game)

def compute_signals(games, uid):
    return {
        "acpl":                  sig_acpl(games, uid),
        "t1_agreement":          sig_t1(games, uid),
        "cpl_std":               sig_cpl_std(games, uid),
        "critical_accuracy":     sig_ca(games, uid),
        "skill_consistency_gap": sig_scg(games, uid),
        "think_time_std":        sig_think_time_std(games, uid),
        "low_acpl_game_rate":    sig_low_acpl_game_rate(games, uid),
    }

SIG_COLS = [
    "acpl", "t1_agreement", "cpl_std",
    "critical_accuracy", "skill_consistency_gap",
    "think_time_std", "low_acpl_game_rate",
]
SIG_NICE = [
    "Avg Centipawn Loss", "T1 Move Agreement", "CPL Std-Dev",
    "Critical Accuracy", "Skill-Consistency Gap",
    "Think-Time Std-Dev", "Low-ACPL Game Rate",
]

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
W  = 66
HR = "═" * W

def main():
    print(f"\n{HR}")
    print("  Panopticon — Real Cheater Discovery & Signal Calibration")
    print(HR)

    # ── 1  Collect opponent usernames from top players ───────────────
    print(f"\n[1/6] Scanning {SCAN_PLAYERS} top blitz players for opponents …")
    top_players = get_top_players(SCAN_PLAYERS)
    all_opponents = set()

    for idx, p in enumerate(top_players):
        print(f"  [{idx+1:>3}/{len(top_players)}] {p:24s}", end="  ", flush=True)
        games = fetch_games_ndjson(p, max_games=GAMES_PER_SCAN)
        opponents = set()
        for g in games:
            try:
                w_id = g["players"]["white"]["user"]["id"].lower()
                b_id = g["players"]["black"]["user"]["id"].lower()
                opp = b_id if w_id == p.lower() else w_id
                opponents.add(opp)
            except KeyError:
                continue
        all_opponents.update(opponents)
        print(f"{len(games):>3} games → {len(opponents):>3} opponents  "
              f"[cumul {len(all_opponents)}]")
        time.sleep(1.2)

    print(f"\n  Total unique opponents: {len(all_opponents)}")

    # ── 2  Batch-check for TOS violations ────────────────────────────
    print("\n[2/6] Checking opponent accounts for TOS violations …")
    opp_list = list(all_opponents)
    user_data = batch_check_users(opp_list)

    banned = []
    for u in user_data:
        if u.get("tosViolation") or u.get("disabled"):
            banned.append(u)

    print(f"  Checked {len(user_data)} accounts")
    print(f"  Found {len(banned)} banned/TOS-violation accounts")

    if not banned:
        print("\n  ERROR: No banned accounts found across all opponents. Exiting.")
        sys.exit(1)

    # Show all banned accounts before T1 filtering
    print(f"\n  {len(banned)} banned accounts found (pre-filter):")
    for b in banned[:80]:
        blitz = b.get("perfs", {}).get("blitz", {})
        r = blitz.get("rating", "?")
        g = blitz.get("games", 0)
        print(f"    {b['id']:28s}  rating={r}  blitz_games={g}")

    # ── 3  Download cheater games, apply T1 filter, compute signals ──
    print(f"\n[3/6] Downloading games, applying T1 ≥ {T1_ENGINE_THRESHOLD} engine filter …")

    cheater_rows = []

    # Sort by blitz game count descending — more data = more reliable signals
    banned.sort(key=lambda u: u.get("perfs", {}).get("blitz", {}).get("games", 0),
                reverse=True)

    n_checked = n_skipped_data = n_skipped_t1 = 0

    for b in banned[:120]:
        uid       = b["id"]
        blitz_cnt = b.get("perfs", {}).get("blitz", {}).get("games", 0)
        if blitz_cnt < 30:
            continue

        n_checked += 1
        print(f"  [{n_checked:>3}] {uid:28s}", end="  ", flush=True)
        games   = fetch_games_ndjson(uid, max_games=200)
        n_evals = sum(1 for g in games if g.get("analysis") and len(g["analysis"]) >= 6)

        if len(games) < 15 or n_evals < 10:
            n_skipped_data += 1
            print(f"skip (only {len(games)} games / {n_evals} evals)")
            time.sleep(1.2)
            continue

        # ── T1 filter: engine cheaters have unusually high top-move agreement
        t1 = sig_t1(games, uid)
        if np.isnan(t1) or t1 < T1_ENGINE_THRESHOLD:
            n_skipped_t1 += 1
            t1_str = f"{t1:.3f}" if not np.isnan(t1) else "n/a"
            print(f"skip (T1={t1_str} < {T1_ENGINE_THRESHOLD} — not engine-assisted)")
            time.sleep(1.2)
            continue

        sigs = compute_signals(games, uid)

        if any(np.isnan(v) for v in sigs.values()):
            print(f"skip (incomplete signals)")
            time.sleep(1.2)
            continue

        avg_rating = float(np.mean([get_rating(g, get_color(g, uid)) for g in games]))
        row = {**sigs, "avg_rating": avg_rating, "num_games": len(games),
               "player": uid, "is_cheater": True}
        cheater_rows.append(row)
        print(f"{len(games):>3} games ({n_evals} evals)  "
              f"t1={sigs['t1_agreement']:.3f}  acpl={sigs['acpl']:.3f}  "
              f"ca={sigs['critical_accuracy']:.3f}  scg={sigs['skill_consistency_gap']:+.3f}")
        time.sleep(1.2)

    print(f"\n  Checked:             {n_checked}")
    print(f"  Skipped (low data):  {n_skipped_data}")
    print(f"  Skipped (T1 filter): {n_skipped_t1}")
    print(f"  Engine cheaters:     {len(cheater_rows)}")

    if len(cheater_rows) < 3:
        print(f"  WARNING: Very few engine cheaters found. "
              f"Try lowering T1_ENGINE_THRESHOLD (currently {T1_ENGINE_THRESHOLD}).")

    # ── 4  Load existing clean player data ───────────────────────────
    print("\n[4/6] Loading existing clean player data …")
    old_csv = os.path.join(os.path.dirname(__file__), "player_signals.csv")
    old_df = pd.read_csv(old_csv)
    clean_df = old_df[~old_df.is_cheater].copy()
    print(f"  {len(clean_df)} clean players from previous analysis")

    # ── 5  Build new dataset ─────────────────────────────────────────
    print("\n[5/6] Building updated dataset …")
    cheater_df = pd.DataFrame(cheater_rows)

    combined = pd.concat([clean_df, cheater_df], ignore_index=True)

    # Save
    out_csv = os.path.join(os.path.dirname(__file__), "player_signals.csv")
    combined.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")
    print(f"    {len(clean_df)} clean  +  {len(cheater_df)} real cheaters  "
          f"=  {len(combined)} total")

    # Print comparison
    print(f"\n  Signal comparison (clean vs real cheater):")
    hdr = f"    {'Signal':<28s} {'Clean μ':>8s} {'Clean σ':>8s}  │  {'Cheat μ':>8s} {'Cheat σ':>8s}  {'Sep.':>6s}"
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))
    for col, nice in zip(SIG_COLS, SIG_NICE):
        cm = clean_df[col].mean()
        cs = clean_df[col].std()
        xm = cheater_df[col].mean()
        xs = cheater_df[col].std()
        # separation in clean std units
        sep = abs(xm - cm) / cs if cs > 0 else 0
        print(f"    {nice:<28s} {cm:>+8.4f} {cs:>8.4f}  │  "
              f"{xm:>+8.4f} {xs:>8.4f}  {sep:>5.1f}σ")

    # ── 6  Update panopticon.html ────────────────────────────────────
    print("\n[6/6] Updating Panopticon dashboard with real cheater data …")

    # Build JS array from cheater data
    cheater_js_rows = []
    for _, r in cheater_df.iterrows():
        vals = [round(r[c], 4) for c in SIG_COLS]
        cheater_js_rows.append("[" + ",".join(str(v) for v in vals) + "]")
    cheater_js = "const CHEATERS=[\n" + ",\n".join(cheater_js_rows) + "\n];"

    # Also rebuild clean JS array to be safe
    clean_js_rows = []
    for _, r in clean_df.iterrows():
        vals = [round(r[c], 4) for c in SIG_COLS]
        clean_js_rows.append("[" + ",".join(str(v) for v in vals) + "]")
    clean_js = "const CLEAN=[\n" + ",\n".join(clean_js_rows) + "\n];"

    # Read panopticon HTML
    html_path = os.path.join(os.path.dirname(__file__), "panopticon.html")
    with open(html_path, "r") as f:
        html = f.read()

    # Replace CLEAN array
    html = re.sub(
        r"const CLEAN=\[.*?\];",
        clean_js,
        html,
        flags=re.DOTALL,
    )

    # Replace CHEATERS array
    html = re.sub(
        r"const CHEATERS=\[.*?\];",
        cheater_js,
        html,
        flags=re.DOTALL,
    )

    # Update legend label
    html = html.replace(
        "Synthetic cheaters (n=30)",
        f"Real banned cheaters (n={len(cheater_df)})",
    )
    html = html.replace(
        "Synth. Cheater (n=30)",
        f"Banned cheater (n={len(cheater_df)})",
    )
    html = html.replace("Synth. Cheater", "Banned Cheater")
    html = html.replace("Synthetic cheater", "Banned cheater")

    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Updated {html_path}")

    # Also update the v2 copy
    v2_path = os.path.join(os.path.dirname(__file__), "panopticon_v2.html")
    with open(v2_path, "w") as f:
        f.write(html)
    print(f"  Updated {v2_path}")

    # ── Visualization ────────────────────────────────────────────────
    print("\n  Generating comparison scatter plot …")

    PAIRS = [(0,1),(0,2),(1,3),(2,3),(5,0),(6,1)]
    clean_arr = clean_df[SIG_COLS].values
    cheat_arr = cheater_df[SIG_COLS].values

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor("#0a0d14")
    fig.suptitle(
        f"Panopticon — Real Cheater Signal Profiles\n"
        f"blue = {len(clean_df)} legitimate players    "
        f"red = {len(cheater_df)} Lichess-banned cheaters",
        color="#e0e6f0", fontsize=14, fontweight="bold", y=0.99,
    )

    for ax, (xi, yi) in zip(axes.flat, PAIRS):
        ax.set_facecolor("#111622")
        ax.scatter(clean_arr[:, xi], clean_arr[:, yi],
                   c="#3b82f6", alpha=0.45, s=45, edgecolors="white",
                   linewidth=0.3, label="Legitimate", zorder=2)
        ax.scatter(cheat_arr[:, xi], cheat_arr[:, yi],
                   c="#ef4444", alpha=0.70, s=65, marker="X",
                   edgecolors="#7f1d1d", linewidth=0.5,
                   label="Banned Cheater", zorder=3)
        ax.set_xlabel(SIG_NICE[xi], color="#8892a8", fontsize=10)
        ax.set_ylabel(SIG_NICE[yi], color="#8892a8", fontsize=10)
        ax.set_title(f"{SIG_NICE[xi]}  vs  {SIG_NICE[yi]}",
                     color="#e0e6f0", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#4e5872", labelsize=9)
        ax.grid(True, alpha=0.12, color="white")
        for spine in ax.spines.values():
            spine.set_color("#1e2538")
        if xi == 0 and yi == 1:
            ax.legend(fontsize=8, loc="best", facecolor="#181e2e",
                      edgecolor="#2e3750", labelcolor="#e0e6f0", framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(os.path.dirname(__file__), "real_cheaters.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_png}")

    print(f"\n{HR}")
    print(f"  Done. Panopticon now uses {len(cheater_df)} real Lichess-banned")
    print(f"  cheater profiles instead of synthetic data.")
    print(f"  Open panopticon.html to test against real cheater clusters.")
    print(HR + "\n")


if __name__ == "__main__":
    main()

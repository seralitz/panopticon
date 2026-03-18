#!/usr/bin/env python3
"""
Synthetic Toggle-Cheater Simulation for Panopticon
═══════════════════════════════════════════════════

1. Downloads a real clean player's blitz games from Lichess (ndjson + evals + clocks)
2. Creates a tampered copy:
   • 30 % of moves in complex positions → replaced with engine-perfect evals (CPL ≈ 0)
   • Simple-position moves left untouched (the "toggle" signature)
   • Think-time on injected moves blended toward median to flatten time–complexity r
3. Computes all 5 Panopticon signals on BOTH original and tampered data
4. Runs anomaly scoring → prints side-by-side verdict
5. Renders a comparison scatter-plot grid
"""

import requests, json, time, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
INJECTION_RATE   = 0.30     # fraction of complex-position moves to inject
TIMING_BLEND     = 0.50     # how much to blend think-time toward median (0=none, 1=flat)
ENGINE_NOISE_CP  = 5        # max centipawn noise on injected moves
MAX_GAMES        = 200      # games to download
RNG_SEED         = 42

# ═══════════════════════════════════════════════════════════════════════
# REFERENCE DATA  (from player_signals.csv)
# ═══════════════════════════════════════════════════════════════════════
SIG_COLS = [
    "acpl", "t1_agreement", "cpl_std",
    "critical_accuracy", "skill_consistency_gap",
    "think_time_std", "low_acpl_game_rate",
]
SIG_SHORT  = ["acpl", "t1", "cpl_std", "ca", "scg", "tts", "lar"]
SIG_NICE   = [
    "Avg Centipawn Loss", "T1 Move Agreement", "CPL Std-Dev",
    "Critical Accuracy", "Skill-Consistency Gap",
    "Think-Time Std-Dev", "Low-ACPL Game Rate",
]
# suspicion direction: +1 = higher is suspicious, -1 = lower is suspicious
SIG_DIR     = np.array([-1, +1, -1, +1, +1, -1, +1])
# weights proportional to empirical σ-separation from find_real_cheaters run
SIG_WEIGHTS = np.array([3.0, 3.2, 3.5, 4.4, 4.8, 2.0, 3.0])

ref_path = os.path.join(os.path.dirname(__file__), "player_signals.csv")
_ref     = pd.read_csv(ref_path)
CLEAN_REF  = _ref[~_ref.is_cheater][SIG_COLS].values  # (n_clean, 7)
CHEAT_REF  = _ref[ _ref.is_cheater][SIG_COLS].values   # (n_cheat, 7)
CLEAN_MEAN = CLEAN_REF.mean(axis=0)
CLEAN_STD  = CLEAN_REF.std(axis=0)
CLEAN_STD[CLEAN_STD < 1e-9] = 1e-9

# ═══════════════════════════════════════════════════════════════════════
# LICHESS API
# ═══════════════════════════════════════════════════════════════════════
API = "https://lichess.org/api"

def fetch_user(username):
    r = requests.get(f"{API}/user/{username}", timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_games(username, max_games=200):
    """Download rated blitz games as list-of-dicts (ndjson)."""
    url = f"{API}/games/user/{username}"
    params = dict(rated="true", perfType="blitz", max=str(max_games),
                  evals="true", clocks="true", opening="false")
    r = requests.get(url, params=params,
                     headers={"Accept": "application/x-ndjson"}, timeout=120)
    r.raise_for_status()
    games = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return games

def choose_player():
    """Pick a top-50 blitz player with lots of analysed games."""
    r = requests.get(f"{API}/player/top/50/blitz", timeout=15)
    r.raise_for_status()
    return [p["id"] for p in r.json()["users"]]

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def get_color(game, uid):
    wid = game["players"]["white"]["user"]["id"].lower()
    return "white" if wid == uid.lower() else "black"

def get_evals(game):
    """Return evals in pawns or None."""
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
    """Return clocks in seconds or None."""
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

# ═══════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION  (identical logic to Panopticon dashboard)
# ═══════════════════════════════════════════════════════════════════════
CPL_CAP        = 3.0
OPENING_SKIP   = 10
OUTLIER_ACPL_T = 0.12


def _cpl_seq(ev, is_w):
    """Off-by-one fix: prepend 0.0 so White's first move is included."""
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
    """Average centipawn loss. Lower = stronger / more suspicious."""
    losses = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, get_color(g, uid) == "white"))
    return float(np.mean(losses)) if losses else np.nan


def sig_t1(games, uid):
    """Fraction of moves matching engine top choice. Skips opening + trivial positions."""
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
                continue           # trivial/forced
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
        ev_ext = [0.0] + ev
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
        ev_ext = [0.0] + ev
        for i in range(len(ev)):
            if i < OPENING_SKIP:
                continue
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            if abs(ev_ext[i]) < 1.5:
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
    return np.array([
        sig_acpl(games, uid),           sig_t1(games, uid),
        sig_cpl_std(games, uid),        sig_ca(games, uid),
        sig_scg(games, uid),            sig_think_time_std(games, uid),
        sig_low_acpl_game_rate(games, uid),
    ])

# ═══════════════════════════════════════════════════════════════════════
# TAMPERING ENGINE
# ═══════════════════════════════════════════════════════════════════════
def tamper_games(games, uid, injection_rate=INJECTION_RATE, rng=None):
    """
    Create a deep-copied set of games where:
      • 30 % of complex-position moves have CPL → ≈ 0  (engine-perfect)
      • Think-time on injected moves blended toward per-game median
    Returns (tampered_games, stats_dict).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    tampered = deepcopy(games)
    st = dict(complex_moves=0, injected_moves=0, games_touched=0,
              timing_adjustments=0)

    for g in tampered:
        analysis = g.get("analysis")
        clocks   = g.get("clocks")
        if not analysis or len(analysis) < 6:
            continue
        if not clocks or len(clocks) < 6:
            continue

        color   = get_color(g, uid)
        is_w    = color == "white"
        n       = min(len(analysis), len(clocks))

        # ── extract centipawn evals for neighbourhood checks ─────────
        evals_cp = []
        for a in analysis:
            if "eval" in a:
                evals_cp.append(a["eval"])
            elif "mate" in a:
                evals_cp.append(10000 if a["mate"] > 0 else -10000)
            else:
                evals_cp.append(0)

        # ── per-game median think-time (centiseconds) ────────────────
        thinks = []
        for i in range(2, n):
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            prev = i - 2
            if prev < len(clocks):
                t = clocks[prev] - clocks[i]
                if t > 0:
                    thinks.append(t)
        med_think = int(np.median(thinks)) if thinks else 0

        touched = False
        # ── inject ───────────────────────────────────────────────────
        for i in range(1, n):
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue

            # neighbourhood complexity in centipawns
            lo = max(0, i - 2)
            hi = min(len(evals_cp), i + 3)
            window = evals_cp[lo:hi]
            complexity_cp = max(window) - min(window)

            if complexity_cp < 100:          # < 1.0 pawn → simple, keep original
                continue

            st["complex_moves"] += 1

            if rng.random() > injection_rate:
                continue                     # not selected

            st["injected_moves"] += 1
            touched = True

            # ── eval injection: near-zero CPL + T1 match ─────────────
            prev_eval = evals_cp[i - 1]
            noise = int(rng.integers(-ENGINE_NOISE_CP, ENGINE_NOISE_CP + 1))
            if is_w:
                new_eval = prev_eval + abs(noise)   # slight improvement for white
            else:
                new_eval = prev_eval - abs(noise)   # slight improvement for black
            # Replace entire entry: removes 'best' field, simulating T1 match
            analysis[i] = {"eval": new_eval}
            evals_cp[i] = new_eval

            # ── timing: snap to median (simulate engine fixed delay) ─
            if i >= 2 and i < len(clocks):
                prev_idx = i - 2
                if prev_idx < len(clocks):
                    orig_think = clocks[prev_idx] - clocks[i]
                    if orig_think > 0 and med_think > 0:
                        # Full snap to median for injected moves — engine delay
                        # is uniform, unlike human think time which varies widely.
                        delta = orig_think - med_think
                        clocks[i] = clocks[i] + delta
                        st["timing_adjustments"] += 1

        if touched:
            st["games_touched"] += 1

    return tampered, st

# ═══════════════════════════════════════════════════════════════════════
# ANOMALY SCORING  (same as Panopticon dashboard)
# ═══════════════════════════════════════════════════════════════════════
def anomaly_score(sigs, rd=50):
    z_raw = (sigs - CLEAN_MEAN) / CLEAN_STD * SIG_DIR
    z = np.maximum(0, z_raw)
    if rd > 100:
        z[4] *= 100 / rd               # RD-dampened skill-consistency gap (index 4)
    # weighted composite: signals with stronger empirical separation count more
    w = SIG_WEIGHTS / SIG_WEIGHTS.sum()
    composite = (z * w).sum()
    score = 100 * (2 / (1 + np.exp(-0.7 * composite)) - 1)
    return np.clip(score, 0, 100), z

def verdict(score, n_evals):
    if n_evals < 30:
        return "INSUFFICIENT DATA"
    if n_evals < 50 and score >= 25:
        return "MONITORING (capped)"
    if score < 25:
        return "CLEAN"
    if score < 55:
        return "MONITORING"
    return "FLAGGED"

# ═══════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════
PAIRS = [
    (0, 1), (0, 2), (1, 3),
    (2, 3), (5, 0), (6, 1),
]

def render(orig, tamp, player_name, out_path="toggle_cheater_test.png"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor("#0a0d14")
    fig.suptitle(
        f"Panopticon Toggle-Cheater Detection Test — {player_name}\n"
        f"green ★ = original (clean)    red ★ = tampered (30 % engine injection)",
        color="#e0e6f0", fontsize=14, fontweight="bold", y=0.99,
    )

    for ax, (xi, yi) in zip(axes.flat, PAIRS):
        ax.set_facecolor("#111622")
        # reference clusters
        ax.scatter(CLEAN_REF[:, xi], CLEAN_REF[:, yi],
                   c="#3b82f6", alpha=0.40, s=40, edgecolors="white",
                   linewidth=0.3, label="Legitimate (n=120)", zorder=2)
        ax.scatter(CHEAT_REF[:, xi], CHEAT_REF[:, yi],
                   c="#ef4444", alpha=0.55, s=55, marker="X",
                   edgecolors="#7f1d1d", linewidth=0.4,
                   label=f"Cheater ref (n={len(CHEAT_REF)})", zorder=3)
        # original player
        ax.scatter([orig[xi]], [orig[yi]],
                   c="#22c55e", s=260, marker="*", edgecolors="white",
                   linewidth=1.2, label="Original (clean)", zorder=5)
        # tampered player
        ax.scatter([tamp[xi]], [tamp[yi]],
                   c="#ef4444", s=260, marker="*", edgecolors="white",
                   linewidth=1.2, label="Tampered (injected)", zorder=5)
        # arrow from original → tampered
        ax.annotate("", xy=(tamp[xi], tamp[yi]),
                    xytext=(orig[xi], orig[yi]),
                    arrowprops=dict(arrowstyle="->", color="#f59e0b",
                                    lw=2, connectionstyle="arc3,rad=.15"))

        ax.set_xlabel(SIG_NICE[xi], color="#8892a8", fontsize=10)
        ax.set_ylabel(SIG_NICE[yi], color="#8892a8", fontsize=10)
        ax.set_title(f"{SIG_NICE[xi]}  vs  {SIG_NICE[yi]}",
                     color="#e0e6f0", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#4e5872", labelsize=9)
        ax.grid(True, alpha=0.12, color="white")
        for spine in ax.spines.values():
            spine.set_color("#1e2538")
        if xi == 0 and yi == 1:
            ax.legend(fontsize=7.5, loc="upper right",
                      facecolor="#181e2e", edgecolor="#2e3750",
                      labelcolor="#e0e6f0", framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Visualisation saved → {out_path}")

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
W  = 62
HR = "═" * W

def main():
    print(f"\n{HR}")
    print("  Panopticon — Toggle-Cheater Detection Test")
    print(HR)

    # ── 1  Pick a player ─────────────────────────────────────────────
    print("\n[1] Selecting a test subject from the Lichess blitz leaderboard …")
    players = choose_player()

    target = None
    for p in players:
        print(f"    Trying {p} …", end=" ", flush=True)
        games = fetch_games(p, MAX_GAMES)
        n_evals = sum(1 for g in games
                      if g.get("analysis") and len(g["analysis"]) >= 6)
        print(f"{len(games)} games, {n_evals} w/ evals", flush=True)
        if n_evals >= 60:
            target = p
            break
        time.sleep(1.2)

    if not target:
        print("  ERROR: no player with ≥ 60 analysed games found.")
        sys.exit(1)

    uid = target.lower()
    user = fetch_user(target)
    rating = user.get("perfs", {}).get("blitz", {}).get("rating", "?")
    rd     = user.get("perfs", {}).get("blitz", {}).get("rd", 50)
    n_evals = sum(1 for g in games
                  if g.get("analysis") and len(g["analysis"]) >= 6)
    n_clocks = sum(1 for g in games
                   if g.get("clocks") and len(g["clocks"]) >= 6)

    print(f"\n  ✓  Selected: {user.get('username', target)}")
    print(f"     Rating {rating} ± {rd} RD   |   "
          f"{len(games)} games ({n_evals} w/ evals, {n_clocks} w/ clocks)")

    # ── 2  Tamper ────────────────────────────────────────────────────
    print(f"\n[2] Injecting engine assistance "
          f"(rate={INJECTION_RATE:.0%}, noise=±{ENGINE_NOISE_CP}cp, "
          f"timing blend={TIMING_BLEND:.0%}) …")

    tampered, st = tamper_games(games, uid)

    print(f"    Complex-position moves found : {st['complex_moves']:,}")
    print(f"    Moves injected (engine-perf) : {st['injected_moves']:,}  "
          f"({st['injected_moves']}/{st['complex_moves']} = "
          f"{st['injected_moves']/max(1,st['complex_moves']):.1%})")
    print(f"    Games touched                : {st['games_touched']:,} / {len(games)}")
    print(f"    Timing adjustments           : {st['timing_adjustments']:,}")

    # ── 3  Compute signals ───────────────────────────────────────────
    print("\n[3] Computing behavioural signals …")

    orig_sigs = compute_signals(games, uid)
    tamp_sigs = compute_signals(tampered, uid)

    # replace NaN with clean mean for safety
    for i in range(5):
        if np.isnan(orig_sigs[i]):
            orig_sigs[i] = CLEAN_MEAN[i]
        if np.isnan(tamp_sigs[i]):
            tamp_sigs[i] = CLEAN_MEAN[i]

    print()
    hdr = f"    {'Signal':<28s} {'Original':>10s} {'Tampered':>10s} {'Δ':>9s}  Direction"
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))
    for i in range(5):
        delta = tamp_sigs[i] - orig_sigs[i]
        # did the signal move toward the cheater cluster?
        cheat_dir = "→ cheater" if delta * SIG_DIR[i] > 0.001 else "—"
        print(f"    {SIG_NICE[i]:<28s} {orig_sigs[i]:>+10.4f} "
              f"{tamp_sigs[i]:>+10.4f} {delta:>+9.4f}  {cheat_dir}")

    # ── 4  Anomaly scoring ───────────────────────────────────────────
    print(f"\n[4] Anomaly scoring (RD={rd}) …")

    orig_score, orig_z = anomaly_score(orig_sigs, rd)
    tamp_score, tamp_z = anomaly_score(tamp_sigs, rd)
    orig_v = verdict(orig_score, n_evals)
    tamp_v = verdict(tamp_score, n_evals)

    print(f"\n    {'':28s} {'Original':>12s}  {'Tampered':>12s}")
    print(f"    {'':28s} {'─'*12:>12s}  {'─'*12:>12s}")
    print(f"    {'Anomaly score':<28s} {orig_score:>11.1f}   {tamp_score:>11.1f}")
    print(f"    {'Verdict':<28s} {orig_v:>12s}  {tamp_v:>12s}")

    print(f"\n    Per-signal suspicion z-scores (tampered):")
    triggered = []
    for i in range(5):
        flag = ""
        if tamp_z[i] >= 2.5:
            flag = "  ◆ ALERT"
            triggered.append(SIG_NICE[i])
        elif tamp_z[i] >= 1.2:
            flag = "  ▸ elevated"
        print(f"      {SIG_NICE[i]:<28s} z = {tamp_z[i]:5.2f}  "
              f"(was {orig_z[i]:5.2f}){flag}")

    # ── 5  Verdict ───────────────────────────────────────────────────
    detected = tamp_v in ("FLAGGED", "MONITORING")
    print(f"\n{'─' * W}")
    if detected:
        marker = "✓ DETECTED" if tamp_v == "FLAGGED" else "⚠ PARTIALLY DETECTED"
        print(f"  {marker}")
        print(f"  Original was {orig_v}, tampered shifted to {tamp_v}.")
        if triggered:
            print(f"  Trigger signals: {', '.join(triggered)}")
    else:
        print(f"  ✗ NOT DETECTED — tampered profile still reads {tamp_v}.")
        print(f"  The injection rate or complexity threshold may need tuning.")
    print(f"{'─' * W}")

    # ── 6  Visualise ─────────────────────────────────────────────────
    print("\n[5] Rendering comparison scatter plots …")
    render(orig_sigs, tamp_sigs, user.get("username", target))

    # ── 7  Detailed per-move audit ───────────────────────────────────
    print("\n[6] Per-move audit sample (first 3 tampered games) …")
    audit_count = 0
    for idx, (orig_g, tamp_g) in enumerate(zip(games, tampered)):
        if audit_count >= 3:
            break
        orig_a = orig_g.get("analysis", [])
        tamp_a = tamp_g.get("analysis", [])
        if len(orig_a) < 6:
            continue
        is_w = get_color(orig_g, uid) == "white"
        changed = []
        for i in range(1, min(len(orig_a), len(tamp_a))):
            mine = (i % 2 == 0) if is_w else (i % 2 == 1)
            if not mine:
                continue
            oe = orig_a[i].get("eval", orig_a[i].get("mate", "?"))
            te = tamp_a[i].get("eval", tamp_a[i].get("mate", "?"))
            if oe != te:
                changed.append((i, oe, te))
        if changed:
            audit_count += 1
            gid = orig_g.get("id", f"game-{idx}")
            print(f"\n    Game {gid}  ({'white' if is_w else 'black'})  "
                  f"— {len(changed)} moves modified:")
            for half, oe, te in changed[:8]:
                move_num = (half // 2) + 1
                side = "." if half % 2 == 0 else "..."
                orig_cp = f"{oe}cp" if isinstance(oe, int) else str(oe)
                tamp_cp = f"{te}cp" if isinstance(te, int) else str(te)
                print(f"      move {move_num}{side}  eval {orig_cp} → {tamp_cp}")
            if len(changed) > 8:
                print(f"      … and {len(changed) - 8} more")

    print(f"\n{HR}")
    print("  Test complete.")
    print(HR + "\n")


if __name__ == "__main__":
    main()

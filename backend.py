#!/usr/bin/env python3
"""
Panopticon Backend
──────────────────
Serves panopticon.html and proxies Lichess API calls, computing
behavioral signals server-side to avoid browser CORS limitations.

Run:  python backend.py
Open: http://localhost:5000
"""

import os
import json
import time
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_file
from tetris import compute_cross_game_signals

# ─── Constants ───────────────────────────────────────────────────────────────
API                 = "https://lichess.org/api"
CPL_CAP             = 3.0
OPENING_SKIP        = 10
OUTLIER_ACPL_T      = 0.12
OUTLIER_Z_THRESHOLD = 2.0

SIG_COLS = [
    "acpl", "cpl_std", "critical_accuracy", "skill_consistency_gap",
    "think_time_std", "low_acpl_game_rate", "outlier_game_count",
]
# Suspicion direction: +1 = higher is suspicious, -1 = lower is suspicious
SIG_DIR     = np.array([-1, -1, +1, +1, -1, +1, +1], dtype=float)
# Weights from empirical σ-separation
SIG_WEIGHTS = np.array([3.0, 3.5, 4.4, 4.8, 2.0, 3.0, 3.5], dtype=float)

# ─── Load clean-player baseline ──────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_csv  = os.path.join(_here, "player_signals.csv")

_df        = pd.read_csv(_csv)
_clean     = _df[~_df.is_cheater][SIG_COLS].values
CLEAN_MEAN = _clean.mean(axis=0)
CLEAN_STD  = _clean.std(axis=0)

# ─── Lichess API helpers ──────────────────────────────────────────────────────

def fetch_user(username):
    r = requests.get(f"{API}/user/{username}", timeout=15)
    if r.status_code == 404:
        return None, f'User "{username}" not found on Lichess'
    r.raise_for_status()
    return r.json(), None


def fetch_games(username, max_games=200):
    url    = f"{API}/games/user/{username}"
    params = dict(rated="true", perfType="blitz", max=str(max_games),
                  evals="true", clocks="true", moves="false", opening="false")
    try:
        r = requests.get(url, params=params,
                         headers={"Accept": "application/x-ndjson"}, timeout=120)
        if r.status_code in (403, 404):
            return [], None
        r.raise_for_status()
    except Exception as e:
        return [], str(e)
    games = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return games, None

# ─── Signal helpers ───────────────────────────────────────────────────────────

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


def _cpl_seq(ev, is_w):
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

# ─── Signal functions ─────────────────────────────────────────────────────────

def sig_acpl(games, uid):
    losses = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses.extend(_cpl_seq(ev, get_color(g, uid) == "white"))
    return float(np.mean(losses)) if losses else np.nan


def sig_cpl_std(games, uid):
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


def sig_outlier_game_count(games, uid):
    per_game = []
    for g in games:
        ev = get_evals(g)
        if not ev:
            continue
        losses = _cpl_seq(ev, get_color(g, uid) == "white")
        if len(losses) >= 5:
            per_game.append(float(np.mean(losses)))
    if len(per_game) < 10:
        return np.nan
    mu    = float(np.mean(per_game))
    sigma = float(np.std(per_game))
    if sigma < 1e-9:
        return 0.0
    return float(sum(1 for a in per_game if (mu - a) / sigma > OUTLIER_Z_THRESHOLD))


def compute_signals(games, uid):
    return {
        "acpl":                  sig_acpl(games, uid),
        "cpl_std":               sig_cpl_std(games, uid),
        "critical_accuracy":     sig_ca(games, uid),
        "skill_consistency_gap": sig_scg(games, uid),
        "think_time_std":        sig_think_time_std(games, uid),
        "low_acpl_game_rate":    sig_low_acpl_game_rate(games, uid),
        "outlier_game_count":    sig_outlier_game_count(games, uid),
    }

# ─── Scoring ──────────────────────────────────────────────────────────────────

def anomaly_score(sigs_arr, rd=50):
    z_raw = (sigs_arr - CLEAN_MEAN) / CLEAN_STD * SIG_DIR
    z     = np.maximum(0, z_raw)
    if rd > 100:
        z[3] *= 100 / rd          # RD-dampened skill-consistency gap (index 3)
    w         = SIG_WEIGHTS / SIG_WEIGHTS.sum()
    composite = float((z * w).sum())
    score     = 100 * (2 / (1 + np.exp(-0.7 * composite)) - 1)
    return float(np.clip(score, 0, 100)), z.tolist()


def verdict(score, eval_count):
    if eval_count < 30:
        return {"label": "INSUFFICIENT_DATA",
                "desc": f"Only {eval_count} evaluated games — need at least 30 for any verdict."}
    if eval_count < 50:
        if score < 25:
            return {"label": "CLEAN",
                    "desc": "Behavioral signals are within normal ranges for a human player."}
        return {"label": "MONITORING",
                "desc": (f"Some signals deviate from typical play, but only {eval_count} evaluated "
                         f"games available (< 50). Verdict capped at MONITORING.")}
    if score < 25:
        return {"label": "CLEAN",
                "desc": "Behavioral signals are within normal ranges for a human player."}
    if score < 55:
        return {"label": "MONITORING",
                "desc": "Some signals deviate from typical human play. May warrant further review."}
    return {"label": "FLAGGED",
            "desc": "Multiple signals show patterns consistent with engine assistance."}

# ─── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return send_file(os.path.join(_here, "panopticon.html"))


@app.route("/api/analyze/<username>")
def analyze(username):
    tetris_user = request.args.get("tetris_user", "").strip()

    # 1. Profile
    user, err = fetch_user(username)
    if err:
        return jsonify({"error": err}), 404

    perf        = user.get("perfs", {}).get("blitz", {})
    rating      = perf.get("rating", "?")
    rd          = perf.get("rd", 150)
    total_games = perf.get("games", 0)

    # 2. Games
    games, err = fetch_games(username)
    if err:
        return jsonify({"error": f"Could not fetch games: {err}"}), 502

    if len(games) < 15:
        return jsonify({"error": f"Only {len(games)} rated blitz games found — need at least 15."}), 422

    uid          = username.lower()
    evals_count  = sum(1 for g in games if g.get("analysis") and len(g["analysis"]) >= 6)
    clocks_count = sum(1 for g in games if g.get("clocks") and len(g["clocks"]) >= 6)

    # 3. Chess signals
    sigs_dict = compute_signals(games, uid)
    sigs_arr  = np.array([sigs_dict[c] for c in SIG_COLS], dtype=float)

    # Replace NaN with clean mean
    for i in range(len(SIG_COLS)):
        if np.isnan(sigs_arr[i]):
            sigs_arr[i] = CLEAN_MEAN[i]

    # 4. Score
    score, z_scores = anomaly_score(sigs_arr, rd)
    v = verdict(score, evals_count)

    response = {
        "meta": {
            "username":       user.get("username", username),
            "rating":         rating,
            "rd":             rd,
            "total_games":    total_games,
            "games_analyzed": len(games),
            "evals_count":    evals_count,
            "clocks_count":   clocks_count,
        },
        "sigs":        sigs_arr.tolist(),
        "z_scores":    z_scores,
        "score":       score,
        "verdict":     v,
        "clean_means": CLEAN_MEAN.tolist(),
        "clean_stds":  CLEAN_STD.tolist(),
    }

    # 5. Cross-game signals (optional — only when tetris_user supplied)
    if tetris_user:
        response["cross_game"] = compute_cross_game_signals(
            games, uid, tetris_user,
            _cpl_seq, get_evals, get_color,
        )

    return jsonify(response)


if __name__ == "__main__":
    n_clean = int((~_df.is_cheater).sum())
    print(f"Panopticon backend — clean baseline: {n_clean} players, {len(SIG_COLS)} signals")
    print("Starting on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)

#!/usr/bin/env python3
"""
TETR.IO Signal Module
─────────────────────
Fetches per-game records from the public TETR.IO API and computes
three behavioral signals for Panopticon's cross-game entropy analysis.

Signals
-------
tetr_time_cv      – Coefficient of variation (std/mean) of 40L sprint times.
                    Normalises for player skill level so a 20s player and a
                    35s player are compared on the same scale. Human CV is
                    typically 0.05–0.20; consistent bots sit below 0.03.

tetr_finesse_std  – Std-dev of per-game finesse faults (suboptimal keystrokes).
                    Humans have game-to-game variance in fault count; bots and
                    TAS replays show near-zero faults with near-zero variance.

ccec              – Cross-Game Entropy Correlation (Spearman r) between
                    normalised daily chess ACPL deviation and normalised daily
                    Tetris sprint-time deviation.  Same brain → same cognitive
                    state → correlated variance.  Engine in chess breaks the
                    correlation: chess variance compresses while Tetris stays
                    human.  Expected human range: r ≈ +0.2 to +0.5.
                    Engine cheater expectation: r ≈ 0.

Design notes
------------
- fetch_tetris_records()  uses /records/40l/recent  (~25 games, full detail).
- fetch_tetris_scoreflow() uses /labs/scoreflow/     (full history, timestamps +
  times only).  Used for CCEC so we have maximum temporal overlap with chess.
- sig_ccec() accepts the chess helper functions as arguments to avoid a
  circular import between tetris.py and backend.py.
"""

import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import requests
from scipy.stats import spearmanr

# ─── Configuration ────────────────────────────────────────────────────────────
TETR_API    = "https://ch.tetr.io/api"
SESSION_HDR = {"User-Agent": "panopticon-research/1.0 (chess anti-cheat study)"}

MIN_RECORDS_FOR_CV      = 10   # completed games needed for tetr_time_cv
MIN_RECORDS_FOR_FINESSE = 10   # completed games needed for tetr_finesse_std
MIN_OVERLAP_DAYS        = 5    # overlapping days needed for CCEC


# ─── API helpers ──────────────────────────────────────────────────────────────

def _get(url, retries=2):
    """GET with retry on 429 and basic error handling."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=SESSION_HDR, timeout=15)
            if r.status_code == 429:
                print("  [TETR.IO rate-limited, waiting 60s]")
                time.sleep(60)
                continue
            return r
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2)


def fetch_tetris_records(username, mode="40l"):
    """
    Fetch recent 40L sprint records from TETR.IO /records endpoint.

    Returns (records, error_or_None).
    Each record dict contains:
        ts             – ISO 8601 timestamp string
        finaltime_ms   – completion time in milliseconds (float)
        pps            – pieces per second
        finesse_faults – suboptimal keystrokes
        perfect_pieces – pieces placed with perfect finesse
        pieces_placed  – total pieces placed
        inputs         – total keypresses
    Only 'clear' (completed) games are included; topouts are discarded.
    """
    url = f"{TETR_API}/users/{username.lower()}/records/{mode}/recent"
    try:
        r = _get(url)
    except Exception as e:
        return [], str(e)

    if r.status_code == 404:
        return [], f'TETR.IO user "{username}" not found'
    try:
        r.raise_for_status()
    except Exception as e:
        return [], f"TETR.IO API error: {e}"

    body = r.json()
    if not body.get("success"):
        msg = body.get("error", {})
        return [], f"TETR.IO error: {msg}"

    records = []
    for e in body.get("data", {}).get("entries", []):
        res   = e.get("results", {})
        # Discard unfinished games
        if res.get("gameoverreason") != "clear":
            continue
        stats = res.get("stats", {})
        agg   = res.get("aggregatestats", {})
        fin   = stats.get("finesse", {})
        records.append({
            "ts":              e.get("ts", ""),
            "finaltime_ms":    stats.get("finaltime", float("nan")),
            "pps":             agg.get("pps", float("nan")),
            "finesse_faults":  fin.get("faults", float("nan")),
            "perfect_pieces":  fin.get("perfectpieces", float("nan")),
            "pieces_placed":   stats.get("piecesplaced", float("nan")),
            "inputs":          stats.get("inputs", float("nan")),
        })
    return records, None


def fetch_tetris_scoreflow(username, mode="40l"):
    """
    Fetch the full historical 40L timeline via the TETR.IO Labs scoreflow endpoint.

    Returns (timeline, error_or_None).
    Each timeline entry: {"date": "YYYY-MM-DD", "finaltime_s": float}

    The scoreflow payload contains every game ever played as compressed
    delta-timestamps.  Crucially it covers the full career, not just recent
    games — giving much deeper temporal overlap with chess history for CCEC.

    Scoreflow format:
        {"startTime": <epoch_ms>, "points": [[delta_ms, pb_flag, neg_ms], ...]}
    """
    url = f"{TETR_API}/labs/scoreflow/{username.lower()}/{mode}"
    try:
        r = _get(url)
    except Exception as e:
        return [], str(e)

    if r.status_code == 404:
        return [], f'TETR.IO user "{username}" not found (scoreflow)'
    try:
        r.raise_for_status()
    except Exception as e:
        return [], f"Scoreflow API error: {e}"

    body = r.json()
    if not body.get("success"):
        return [], f"Scoreflow error: {body.get('error', 'unknown')}"

    data     = body.get("data", {})
    start_ms = data.get("startTime", 0)
    points   = data.get("points", [])

    timeline = []
    for pt in points:
        if len(pt) < 3:
            continue
        delta_ms, _pb, neg_ms = pt
        epoch_ms    = start_ms + delta_ms
        finaltime_s = -neg_ms / 1000.0            # negated ms → positive seconds
        date_str    = datetime.fromtimestamp(
            epoch_ms / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        timeline.append({"date": date_str, "finaltime_s": finaltime_s})
    return timeline, None


# ─── TETR.IO-side signals ─────────────────────────────────────────────────────

def sig_tetr_time_cv(records):
    """
    Coefficient of variation (std / mean) of 40L completion times in seconds.

    Why CV over raw std:
        A world-class player averaging 18s with std 0.5s has CV=0.028 (suspicious).
        A casual player averaging 80s with std 2s   has CV=0.025 (similar suspicion).
        Raw std would rate the casual player as "more variable" and miss the
        world-class bot — CV normalises for skill level.

    Approximate human reference ranges (from TETR.IO community data):
        Sub-30s players:  CV ≈ 0.04–0.12
        30–60s players:   CV ≈ 0.07–0.18
        60s+ players:     CV ≈ 0.10–0.25
        Bots/TAS:         CV < 0.02
    """
    times = [
        r["finaltime_ms"] / 1000.0
        for r in records
        if not np.isnan(r.get("finaltime_ms", float("nan")))
    ]
    if len(times) < MIN_RECORDS_FOR_CV:
        return float("nan")
    mean = float(np.mean(times))
    if mean < 1e-9:
        return float("nan")
    return float(np.std(times)) / mean


def sig_tetr_finesse_std(records):
    """
    Std-dev of per-game finesse fault counts.

    Finesse faults = keystrokes beyond the theoretical minimum needed to
    place each piece optimally.  A fault count of 0 across all games is
    physically possible only for automated play; humans always vary.

    Two-dimensional suspicious pattern:
        Low mean  + low std  → consistently perfect  → bot
        High mean + high std → consistently imperfect → human (normal)
        Low mean  + high std → sometimes perfect, sometimes not → human warming up

    Only the first pattern (mean AND std near zero) is engine-suspicious.
    This function returns std only; pair with mean from records for full picture.
    """
    faults = [
        r["finesse_faults"]
        for r in records
        if not np.isnan(r.get("finesse_faults", float("nan")))
    ]
    if len(faults) < MIN_RECORDS_FOR_FINESSE:
        return float("nan")
    return float(np.std(faults))


# ─── Cross-game daily alignment ───────────────────────────────────────────────

def _chess_daily_acpl(chess_games, chess_uid, cpl_seq_fn, get_evals_fn, get_color_fn):
    """
    Group chess games by UTC date; compute mean ACPL per day.
    Returns {date_str: mean_acpl}.  Only days with ≥1 evaluated game included.

    Parameters
    ----------
    chess_games   : list of Lichess game dicts (from backend fetch_games)
    chess_uid     : Lichess username (lowercase)
    cpl_seq_fn    : backend._cpl_seq
    get_evals_fn  : backend.get_evals
    get_color_fn  : backend.get_color
    """
    by_day = defaultdict(list)
    for g in chess_games:
        created_ms = g.get("createdAt")
        if not created_ms:
            continue
        date = datetime.fromtimestamp(
            created_ms / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        ev = get_evals_fn(g)
        if not ev:
            continue
        losses = cpl_seq_fn(ev, get_color_fn(g, chess_uid) == "white")
        if losses:
            by_day[date].append(float(np.mean(losses)))
    return {d: float(np.mean(v)) for d, v in by_day.items()}


def _tetris_daily_sprint(scoreflow):
    """
    Group TETR.IO scoreflow entries by UTC date; compute mean sprint time per day.
    Returns {date_str: mean_sprint_seconds}.
    """
    by_day = defaultdict(list)
    for pt in scoreflow:
        by_day[pt["date"]].append(pt["finaltime_s"])
    return {d: float(np.mean(v)) for d, v in by_day.items()}


# ─── Cross-game entropy correlation ──────────────────────────────────────────

def sig_ccec(chess_games, chess_uid, scoreflow,
             cpl_seq_fn, get_evals_fn, get_color_fn):
    """
    Cross-Game Entropy Correlation (CCEC) — Spearman r between daily
    chess performance deviation and daily Tetris performance deviation.

    Hypothesis
    ----------
    A human's cognitive state (fatigue, focus, emotional arousal) originates
    from a single brain and manifests across all cognitive tasks simultaneously.
    A good chess day should correlate with a good Tetris day because the same
    neurological substrate drives both.

    An engine-assisted chess player breaks this coupling: their chess
    performance is held artificially constant by the engine while their
    Tetris performance varies naturally — producing a near-zero correlation.

    Method
    ------
    For each day with ≥1 evaluated chess game AND ≥1 completed Tetris sprint:

        chess_dev(day)  = -(acpl(day) − μ_acpl)  / σ_acpl
        tetris_dev(day) = -(time(day) − μ_time) / σ_time

    Negated so positive = good day for both games.
    CCEC = Spearman_r(chess_dev_series, tetris_dev_series)

    Returns
    -------
    (r, pvalue, n_overlap_days, chess_series, tetris_series)

    chess_series  : list of [date_str, acpl_float]   (all chess days, for chart)
    tetris_series : list of [date_str, sprint_float] (all tetris days, for chart)

    r and pvalue are NaN when n_overlap_days < MIN_OVERLAP_DAYS.
    """
    chess_by_day  = _chess_daily_acpl(chess_games, chess_uid,
                                      cpl_seq_fn, get_evals_fn, get_color_fn)
    tetris_by_day = _tetris_daily_sprint(scoreflow)

    chess_series  = [[d, v] for d, v in sorted(chess_by_day.items())]
    tetris_series = [[d, v] for d, v in sorted(tetris_by_day.items())]

    common = sorted(set(chess_by_day) & set(tetris_by_day))
    n_overlap = len(common)

    if n_overlap < MIN_OVERLAP_DAYS:
        return float("nan"), float("nan"), n_overlap, chess_series, tetris_series

    chess_vals  = np.array([chess_by_day[d]  for d in common])
    tetris_vals = np.array([tetris_by_day[d] for d in common])

    # Normalise to per-player deviation: positive = good day
    def _dev(arr):
        sigma = arr.std()
        if sigma < 1e-9:
            return np.zeros_like(arr)
        return -(arr - arr.mean()) / sigma   # negate: lower ACPL/time = better

    r, pval = spearmanr(_dev(chess_vals), _dev(tetris_vals))
    return float(r), float(pval), n_overlap, chess_series, tetris_series


# ─── Convenience: compute all three at once ──────────────────────────────────

def compute_cross_game_signals(chess_games, chess_uid, tetris_username,
                                cpl_seq_fn, get_evals_fn, get_color_fn):
    """
    Fetch TETR.IO data for tetris_username and compute all three cross-game signals.

    Returns dict:
    {
        "available":       bool,
        "error":           str or None,
        "tetris_user":     str,
        "records_count":   int,
        "tetr_time_cv":    float or None,
        "tetr_finesse_std":float or None,
        "ccec":            float or None,
        "ccec_pvalue":     float or None,
        "overlapping_days":int,
        "chess_daily":     [[date, acpl], ...],
        "tetris_daily":    [[date, sprint_s], ...],
    }
    """
    result = {
        "available":        False,
        "error":            None,
        "tetris_user":      tetris_username,
        "records_count":    0,
        "tetr_time_cv":     None,
        "tetr_finesse_std": None,
        "ccec":             None,
        "ccec_pvalue":      None,
        "overlapping_days": 0,
        "chess_daily":      [],
        "tetris_daily":     [],
    }

    # Detailed records for per-game variance signals
    records, err = fetch_tetris_records(tetris_username)
    if err:
        result["error"] = err
        return result

    result["records_count"] = len(records)

    tcv   = sig_tetr_time_cv(records)
    tfstd = sig_tetr_finesse_std(records)
    result["tetr_time_cv"]     = None if np.isnan(tcv)   else round(tcv,   6)
    result["tetr_finesse_std"] = None if np.isnan(tfstd) else round(tfstd, 4)

    # Full timeline for CCEC (uses scoreflow for maximum historical depth)
    scoreflow, serr = fetch_tetris_scoreflow(tetris_username)
    if serr:
        # Scoreflow failed; fall back to constructing timeline from records
        scoreflow = []
        for rec in records:
            ts_str = rec["ts"][:10] if rec["ts"] else None
            if ts_str and not np.isnan(rec["finaltime_ms"]):
                scoreflow.append({
                    "date":        ts_str,
                    "finaltime_s": rec["finaltime_ms"] / 1000.0,
                })

    r, pval, n_overlap, chess_series, tetris_series = sig_ccec(
        chess_games, chess_uid, scoreflow,
        cpl_seq_fn, get_evals_fn, get_color_fn,
    )

    result["available"]        = True
    result["ccec"]             = None if np.isnan(r)    else round(r,    4)
    result["ccec_pvalue"]      = None if np.isnan(pval) else round(pval, 4)
    result["overlapping_days"] = n_overlap
    result["chess_daily"]      = chess_series
    result["tetris_daily"]     = tetris_series

    return result

# Chess Cheater Detection

A behavioral analysis system for detecting chess engine usage on Lichess. Analyzes player move quality, timing patterns, and cross-game consistency to flag suspicious accounts.
5.9σ separation on confirmed banned accounts

## How It Works

The system downloads rated blitz games from the Lichess API and computes 8 behavioral signals per player:

| Signal | Description | Why It Matters |
|--------|-------------|----------------|
| **ACPL** | Average centipawn loss | Engine users average 0.02–0.08 cp; GMs average 0.15–0.25 cp |
| **T1 Agreement** | Fraction of moves matching engine's top choice | Unnaturally high for cheaters |
| **CPL Std-Dev** | Variance in move quality | Engines are consistent; humans are erratic |
| **Critical Accuracy** | Accuracy in complex positions (>1 pawn swing) | Cheaters maintain precision under pressure |
| **Skill Consistency Gap** | Hard-move accuracy vs. rating-expected accuracy | Cheaters outperform their own rating in hard positions |
| **Think-Time Std-Dev** | Variance in move timing | Engine-assisted play shows robotic uniformity |
| **Low-ACPL Game Rate** | % of games below 0.12 cp threshold | Sustained perfection across many games is suspicious |

Each signal is Z-scored against a clean-player baseline, weighted by empirical separation power, and combined into a 0–100 anomaly score.

**Verdicts:**
- `CLEAN` — score < 25
- `MONITORING` — score 25–55
- `FLAGGED` — score > 55 (with sufficient data)
- `INSUFFICIENT_DATA` — fewer than 30 evaluated games

### Cross-Game Entropy Correlation

An optional analysis correlates daily chess performance variance with TETR.IO (Tetris) sprint times. Humans show correlated variance across different games; engine users break this coupling.

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3, Flask, NumPy, Pandas, SciPy, Matplotlib, Requests

## Usage

### Web Interface (Real-Time Player Lookup)

```bash
python backend.py
```

Starts a Flask server at `http://localhost:8080`. Enter any Lichess username to get a behavioral analysis and anomaly score. Optionally link a TETR.IO profile for cross-game entropy analysis.

### Batch Dataset Generation

```bash
python cheater_detection.py
```

Downloads ~10,000 games from the top 50 Lichess blitz players, computes all signals, generates synthetic cheater profiles for comparison, and saves:
- `player_signals.csv` — full signal dataset
- `cheater_detection.png` — scatter plot visualization

### Real Cheater Analysis

```bash
python find_real_cheaters.py
```

Identifies banned Lichess accounts and analyzes their actual behavioral patterns (rather than synthetic ones).

### Synthetic Injection Test

```bash
python toggle_cheater_test.py
```

Downloads a clean player's games, injects engine moves in 30% of complex positions, and verifies the system detects the tampering.

## Project Structure

```
chess_cheater_detection/
├── cheater_detection.py    # Core signal computation and batch analysis
├── backend.py              # Flask API and anomaly scoring
├── tetris.py               # Cross-game entropy correlation (TETR.IO)
├── find_real_cheaters.py   # Real banned-player analysis
├── toggle_cheater_test.py  # Synthetic injection testing
├── panopticon.html         # Web UI
├── panopticon_v2.html      # Alternative web UI
├── player_signals.csv      # Baseline signal dataset
└── requirements.txt
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GAMES_PER_PLAYER` | 400 | Max games analyzed per player |
| `OPENING_SKIP` | 10 | Half-moves skipped (opening book) |
| `CPL_CAP` | 3.0 | Cap on individual centipawn loss (noise reduction) |
| `OUTLIER_ACPL_T` | 0.12 | Threshold for a suspiciously good game |
| `OUTLIER_Z_THRESHOLD` | 2.0 | Z-score for outlier game detection |

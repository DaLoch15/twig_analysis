"""
Simulation Check #4: Multi-Tournament Backtest
===============================================

Runs the Monte Carlo simulation across many historical tournaments
and evaluates aggregate calibration.

WHY MULTI-TOURNAMENT?
    A single tournament's Brier scores are noisy — one event is 147
    player-predictions, and whether the actual winner happened to be
    your #2 or #30 prediction is largely luck. Across 10+ tournaments,
    the metrics stabilize and reveal whether the model genuinely
    extracts signal.

WHAT THIS SCRIPT DOES:
    1. Load data, discover all valid tournaments for backtesting
    2. Filter to full-field events with finish position data
    3. Run backtest() across selected events
    4. Print aggregate metrics + calibration table
    5. Save detailed results for further analysis

RUNTIME ESTIMATE:
    50k sims × ~10 tournaments ≈ 2-5 minutes total
"""

import numpy as np
import pandas as pd
from pathlib import Path

from simulation import (
    backtest,
    prepare_field,
    brier_score,
    brier_skill_score,
    calibration_table,
)

DATA_DIR = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA AND DISCOVER TEST EVENTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Discovering Valid Test Events")
print("=" * 70)

parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
csv_path = DATA_DIR / "processed" / "master_rounds.csv"

if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
elif csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError("Run Phases 0-2 first")

print(f"  Loaded {len(df):,} rows")

# ── Find tournaments suitable for backtesting ──
# Requirements:
#   - Year 2024 or 2025 (enough training history, recent enough to matter)
#   - At least 80 players (skip invitationals and team events)
#   - finish_pos populated for at least 50% of players (need actuals)
#   - predicted_skill populated for at least 80% of players

tourn_stats = df.groupby(["event_id", "event_name", "calendar_year"]).agg(
    n_players=("dg_id", "nunique"),
    skill_pct=("predicted_skill", lambda x: x.notna().mean()),
    finish_pct=("finish_pos", lambda x: x.notna().mean()),
    has_winner=("finish_pos", lambda x: (x == 1).any()),
).reset_index()

# Apply filters
valid = tourn_stats[
    (tourn_stats["calendar_year"].isin([2024, 2025]))
    & (tourn_stats["n_players"] >= 80)
    & (tourn_stats["skill_pct"] >= 0.80)
    & (tourn_stats["finish_pct"] >= 0.40)
    & (tourn_stats["has_winner"] == True)
].sort_values(["calendar_year", "event_name"])

print(f"\n  Found {len(valid)} valid tournaments for backtesting:")
print(f"  {'Event':<45s} {'Year':>5s} {'Players':>8s} {'Fin%':>6s}")
print(f"  {'─' * 68}")
for _, row in valid.iterrows():
    print(f"  {str(row['event_name'])[:45]:<45s} "
          f"{row['calendar_year']:>5.0f} "
          f"{row['n_players']:>8.0f} "
          f"{row['finish_pct']:>5.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD THE TEST EVENT LIST
# ══════════════════════════════════════════════════════════════════════════════
#
# We want a MIX of tournament types:
#   - Full-field regular events (150+ players) — the bread and butter
#   - Smaller elevated events (70-120 players) — stronger fields
#   - At least one major if available
#
# Cap at ~15 tournaments to keep runtime reasonable (~5 min total).
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 2: Selecting Test Events")
print("=" * 70)

# Take up to 15 events, prioritizing variety
# Split roughly evenly between 2024 and 2025 if both available
events_2024 = valid[valid["calendar_year"] == 2024]
events_2025 = valid[valid["calendar_year"] == 2025]

selected = pd.concat([
    events_2024.head(8),
    events_2025.head(7),
]).head(15)

test_events = [
    (row["event_id"], int(row["calendar_year"]))
    for _, row in selected.iterrows()
]

print(f"  Selected {len(test_events)} tournaments for backtesting:")
for eid, yr in test_events:
    name = selected[
        (selected["event_id"] == eid) & (selected["calendar_year"] == yr)
    ]["event_name"].iloc[0]
    print(f"    {name} ({yr})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
#
# 50k sims per tournament (half of the single-tournament test).
# This gives ~0.02% precision on win probabilities — good enough
# for aggregate calibration, and keeps total runtime under 5 minutes.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 3: Running Backtest")
print("=" * 70)

results = backtest(df, test_events, n_sims=50_000, seed=42)

if len(results) == 0:
    print("\n  ⚠ No results! Check that test events have valid data.")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: ADDITIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# Beyond the aggregate metrics that backtest() prints, we want:
#   - Per-tournament breakdown (did some events score much worse?)
#   - Where did actual winners rank in our predictions?
#   - Calibration table for make-cut (our weakest metric last time)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 4: Detailed Analysis")
print("=" * 70)

# ── Per-tournament winner analysis ──
print(f"\n  ── Where Did Actual Winners Rank? ──")
print(f"  {'Event':<40s} {'Winner':<25s} {'PredWin':>8s} {'Rank':>5s}")
print(f"  {'─' * 82}")

for (eid, yr) in test_events:
    event_results = results[
        (results["event_id"] == eid) & (results["year"] == yr)
    ]
    if len(event_results) == 0:
        continue

    winner_row = event_results[event_results["actual_win"] == 1]
    if len(winner_row) == 0:
        continue

    w = winner_row.iloc[0]
    # Rank = how many players had higher or equal win probability
    rank = int((event_results["win"] >= w["win"]).sum())
    event_name = str(w.get("event_name", ""))[:40]
    player_name = str(w.get("player_name", ""))[:25]

    print(f"  {event_name:<40s} {player_name:<25s} "
          f"{w['win']:>7.1%} {'#' + str(rank):>5s}")

# ── Per-tournament Brier Skill Score ──
print(f"\n  ── Per-Tournament Win BSS ──")
print(f"  {'Event':<45s} {'N':>5s} {'BSS':>8s}")
print(f"  {'─' * 62}")

for (eid, yr) in test_events:
    event_results = results[
        (results["event_id"] == eid) & (results["year"] == yr)
    ]
    if len(event_results) == 0:
        continue

    valid_mask = event_results[["win", "actual_win"]].notna().all(axis=1)
    ev = event_results[valid_mask]
    if len(ev) == 0:
        continue

    bss = brier_skill_score(ev["win"], ev["actual_win"])
    event_name = str(ev["event_name"].iloc[0])[:45]
    print(f"  {event_name:<45s} {len(ev):>5d} {bss:>+7.3f}")

# ── Make-cut calibration (our weakest metric) ──
print(f"\n  ── Make-Cut Calibration ──")
cut_valid = results[["make_cut", "actual_cut"]].dropna()
if len(cut_valid) > 0:
    cut_cal = calibration_table(
        cut_valid["make_cut"], cut_valid["actual_cut"],
        bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
              (0.6, 0.8), (0.8, 1.0)],
    )
    print(f"  {'Bucket':<12s} {'N':>6s} {'Predicted':>10s} {'Actual':>10s} {'Diff':>8s}")
    print(f"  {'─' * 48}")
    for _, row in cut_cal.iterrows():
        print(f"  {row['bucket']:<12s} {row['n']:>6d} "
              f"{row['predicted_avg']:>10.1%} {row['actual_rate']:>10.1%} "
              f"{row['diff']:>+7.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 5: Saving Results")
print("=" * 70)

out_dir = DATA_DIR / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(out_dir / "backtest_results.csv", index=False)
print(f"  ✓ Saved backtest_results.csv ({len(results):,} player-events)")

# ── Final summary ──
print(f"\n{'=' * 70}")
print("BACKTEST COMPLETE")
print("=" * 70)
print(f"  Tournaments tested:    {len(test_events)}")
print(f"  Player-events:         {len(results):,}")
print(f"  Simulations per event: 50,000")

# Compute overall BSS one more time for the summary
for outcome, pred_col, actual_col in [
    ("Win", "win", "actual_win"),
    ("Top 5", "top_5", "actual_top5"),
    ("Top 10", "top_10", "actual_top10"),
    ("Top 20", "top_20", "actual_top20"),
    ("Make Cut", "make_cut", "actual_cut"),
]:
    v = results[[pred_col, actual_col]].dropna()
    if len(v) > 0:
        bss = brier_skill_score(v[pred_col], v[actual_col])
        print(f"  {outcome:<12s} BSS: {bss:+.4f} {'✓' if bss > 0 else '✗'}")

print(f"\n  Key interpretation:")
print(f"    All BSS > 0 → model adds value across tournaments")
print(f"    Any BSS < 0 → investigate that outcome category")
print(f"    BSS 0.05-0.15 → good for golf")
print(f"    BSS > 0.15 → excellent (approaching Data Golf quality)")
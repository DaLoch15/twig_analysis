import numpy as np
import pandas as pd
from pathlib import Path

# Import the simulation module — adjust path if needed
from simulation import simulate_tournament, brier_score, brier_skill_score, log_loss

DATA_DIR = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND INSPECT
# ══════════════════════════════════════════════════════════════════════════════
#
# Before we simulate anything, we need to understand what's in the data.
# Key questions:
#   - Which columns does master_rounds actually have?
#   - Which years/tournaments have predicted_skill populated?
#   - Are there any NaN issues we need to handle?
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Loading and Inspecting Data")
print("=" * 70)

parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
csv_path = DATA_DIR / "processed" / "master_rounds.csv"

if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded from parquet: {len(df):,} rows")
elif csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"  Loaded from CSV: {len(df):,} rows")
else:
    raise FileNotFoundError(
        "No master_rounds file found. Run Phases 0-2 first.\n"
        "  python data_pipeline.py\n"
        "  python course_adjustment.py\n"
        "  python skill_estimation.py"
    )

# ── Check which columns we have ──
required_cols = ["predicted_skill", "estimated_sd", "adj_sg", "event_id",
                 "event_name", "calendar_year", "dg_id", "player_name",
                 "round_num"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    print(f"\n  ⚠ Missing columns: {missing}")
    print(f"  Available columns: {sorted(df.columns.tolist())}")
    print(f"\n  If 'predicted_skill' is missing, run Phase 2:")
    print(f"    python skill_estimation.py")
    raise ValueError(f"Missing required columns: {missing}")
else:
    print(f"  ✓ All required columns present")

# ── Check predicted_skill coverage ──
has_skill = df["predicted_skill"].notna()
print(f"\n  Rows with predicted_skill: {has_skill.sum():,} / {len(df):,} "
      f"({has_skill.mean():.1%})")

skill_by_year = df.groupby("calendar_year")["predicted_skill"].apply(
    lambda x: f"{x.notna().mean():.0%}"
)
print(f"\n  Skill coverage by year:")
for year, pct in skill_by_year.items():
    print(f"    {year}: {pct}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PICK A TEST TOURNAMENT
# ══════════════════════════════════════════════════════════════════════════════
#
# We want a tournament that:
#   - Has predicted_skill for most/all players (skip early years if sparse)
#   - Has a reasonable field size (100+ players, not a 30-man invitational)
#   - Has finish_pos data so we can check against actual outcomes
#   - Is in a RECENT year (2024 or 2025) so predictions are based on
#     substantial training history
#
# We'll list the best candidates and pick one.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 2: Finding a Good Test Tournament")
print("=" * 70)

# Summarize tournaments in recent years
recent = df[df["calendar_year"] >= 2023].copy()

tourn_summary = recent.groupby(["event_id", "event_name", "calendar_year"]).agg(
    n_players=("dg_id", "nunique"),
    n_rounds=("round_num", "count"),
    skill_coverage=("predicted_skill", lambda x: x.notna().mean()),
    has_finish=("finish_pos", lambda x: x.notna().mean()),
    avg_skill=("predicted_skill", "mean"),
).reset_index()

# Filter to good candidates
good = tourn_summary[
    (tourn_summary["n_players"] >= 50)
    & (tourn_summary["skill_coverage"] >= 0.8)
].sort_values(["calendar_year", "event_name"], ascending=[False, True])

print(f"\n  Candidate tournaments (≥50 players, ≥80% skill coverage):")
print(f"  {'Event':<40s} {'Year':>5s} {'Players':>8s} {'Skill%':>7s} {'Finish%':>8s}")
print(f"  {'─' * 72}")

for _, row in good.head(20).iterrows():
    print(f"  {str(row['event_name'])[:40]:<40s} "
          f"{row['calendar_year']:>5.0f} "
          f"{row['n_players']:>8.0f} "
          f"{row['skill_coverage']:>6.0%} "
          f"{row['has_finish']:>7.0%}")

# ── Auto-select: pick the most recent large-field event with full data ──
# Prefer a "normal" full-field event (not Tour Championship or team event)
best = good[good["n_players"] >= 100]
if len(best) == 0:
    best = good  # fall back to any good tournament

if len(best) == 0:
    print("\n  ⚠ No suitable tournaments found!")
    print("  This likely means Phase 2 hasn't been run, or the data is very sparse.")
    raise SystemExit(1)

# Pick the most recent one
target = best.iloc[0]
TARGET_EVENT_ID = target["event_id"]
TARGET_YEAR = int(target["calendar_year"])
TARGET_NAME = target["event_name"]

print(f"\n  ► Selected: {TARGET_NAME} ({TARGET_YEAR})")
print(f"    Event ID: {TARGET_EVENT_ID}")
print(f"    Players: {target['n_players']:.0f}")
print(f"    Skill coverage: {target['skill_coverage']:.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: EXTRACT THE FIELD
# ══════════════════════════════════════════════════════════════════════════════
#
# For each player in this tournament, we need their PRE-TOURNAMENT skill
# estimate. That's the predicted_skill value from their FIRST round in
# the event — which was computed using only data available before the
# tournament started (because of the walk-forward feature engineering
# in Phase 2).
#
# We also extract actual outcomes for comparison.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 3: Extracting Tournament Field")
print("=" * 70)

event_mask = (
    (df["event_id"] == TARGET_EVENT_ID)
    & (df["calendar_year"] == TARGET_YEAR)
)
event_df = df[event_mask].copy()

print(f"  Rows for this event: {len(event_df):,}")
print(f"  Unique players: {event_df['dg_id'].nunique()}")

# ── Get pre-tournament skill for each player ──
# Use Round 1 data (this was computed from pre-tournament history only)
field = event_df.groupby(["dg_id", "player_name"]).agg(
    predicted_skill=("predicted_skill", "first"),    # R1 prediction
    estimated_sd=("estimated_sd", "first"),           # R1 variance
    rounds_played=("rounds_played", "first"),         # pre-tourn rounds count
    finish_pos=("finish_pos", "first"),               # actual finish
    made_cut=("made_cut", "first"),                   # actual cut result
    n_rounds_played=("round_num", "count"),           # rounds in this event
).reset_index()

# ── Handle missing values ──
n_missing_skill = field["predicted_skill"].isna().sum()
n_missing_sd = field["estimated_sd"].isna().sum()

if n_missing_skill > 0:
    print(f"\n  ⚠ {n_missing_skill} players missing predicted_skill → using tour prior (-2.0)")
    field["predicted_skill"] = field["predicted_skill"].fillna(-2.0)

if n_missing_sd > 0:
    print(f"  ⚠ {n_missing_sd} players missing estimated_sd → using tour avg (2.9)")
    field["estimated_sd"] = field["estimated_sd"].fillna(2.9)

# ── Check if rounds_played column exists ──
if "rounds_played" not in event_df.columns:
    # Fall back: estimate from golf_time or player_total_rounds
    if "golf_time" in event_df.columns:
        field["rounds_played"] = event_df.groupby("dg_id")["golf_time"].first().values
    else:
        field["rounds_played"] = 50  # default

# ── Print field summary ──
print(f"\n  Field: {len(field)} players")
print(f"\n  Skill distribution:")
print(f"    Max:    {field['predicted_skill'].max():+.3f}  "
      f"({field.loc[field['predicted_skill'].idxmax(), 'player_name']})")
print(f"    Mean:   {field['predicted_skill'].mean():+.3f}")
print(f"    Median: {field['predicted_skill'].median():+.3f}")
print(f"    Min:    {field['predicted_skill'].min():+.3f}")
print(f"    Std:    {field['predicted_skill'].std():.3f}")

print(f"\n  Variance distribution:")
print(f"    Mean SD:   {field['estimated_sd'].mean():.3f}")
print(f"    Min SD:    {field['estimated_sd'].min():.3f}")
print(f"    Max SD:    {field['estimated_sd'].max():.3f}")

print(f"\n  Top 10 pre-tournament skill estimates:")
print(f"  {'Player':<30s} {'Skill':>7s} {'SD':>6s} {'Rounds':>7s}")
print(f"  {'─' * 55}")
for _, row in field.nlargest(10, "predicted_skill").iterrows():
    rp = row.get("rounds_played", "?")
    rp_str = f"{rp:.0f}" if isinstance(rp, (int, float)) and not pd.isna(rp) else "?"
    print(f"  {str(row['player_name'])[:30]:<30s} "
          f"{row['predicted_skill']:>+6.3f} "
          f"{row['estimated_sd']:>5.2f}  "
          f"{rp_str:>6s}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RUN THE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Now we run the Monte Carlo simulation with the extracted field.
# We use 100k simulations for good precision.
# cut_size=65 is standard PGA Tour.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 4: Running Monte Carlo Simulation")
print("=" * 70)

N_SIMS = 100_000

probs, sim_data = simulate_tournament(
    player_means=field["predicted_skill"].values,
    player_sds=field["estimated_sd"].values,
    n_sims=N_SIMS,
    cut_size=65,
    apply_pressure=True,
    player_ids=field["dg_id"].values,
    player_names=field["player_name"].values,
    seed=42,
)

# ── Print predicted probabilities ──
print(f"\n  {'Player':<30s} {'Win':>7s} {'Top5':>7s} {'Top10':>7s} "
      f"{'Top20':>7s} {'MkCut':>7s} {'E[Fin]':>7s}")
print(f"  {'─' * 80}")

for _, row in probs.head(25).iterrows():
    name = str(row.get("player_name", f"ID:{row['player_id']}"))[:30]
    print(f"  {name:<30s} "
          f"{row['win']:>6.1%} {row['top_5']:>6.1%} {row['top_10']:>6.1%} "
          f"{row['top_20']:>6.1%} {row['make_cut']:>6.1%} "
          f"{row['expected_finish']:>7.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: COMPARE TO ACTUAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the moment of truth. We merge our predictions with what
# actually happened and check:
#   1. Did the actual winner have a reasonably high win probability?
#   2. Do make-cut predictions match actual cut outcomes?
#   3. Are Brier scores better than a naive baseline?
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 5: Comparing Predictions to Actual Results")
print("=" * 70)

# ── Merge predictions with actuals ──
merged = probs.merge(
    field[["dg_id", "player_name", "finish_pos", "made_cut"]],
    left_on="player_id",
    right_on="dg_id",
    how="inner",
    suffixes=("_pred", "_actual"),
)

# Use the correct player_name column
name_col = "player_name_actual" if "player_name_actual" in merged.columns else "player_name"

# ── Compute actual binary outcomes ──
merged["actual_win"] = (merged["finish_pos"] == 1).astype(float)
merged["actual_top5"] = (merged["finish_pos"] <= 5).astype(float)
merged["actual_top10"] = (merged["finish_pos"] <= 10).astype(float)
merged["actual_top20"] = (merged["finish_pos"] <= 20).astype(float)
merged["actual_cut"] = merged["made_cut"].astype(float)

# Handle NaN finish positions (for MC players with no numeric finish)
# MC players didn't win or finish top-N
for col in ["actual_win", "actual_top5", "actual_top10", "actual_top20"]:
    merged[col] = merged[col].fillna(0.0)
merged["actual_cut"] = merged["actual_cut"].fillna(0.0)

n_matched = len(merged)
print(f"\n  Matched {n_matched} players between predictions and actuals")

# ── Who actually won? ──
actual_winner = merged[merged["actual_win"] == 1]
if len(actual_winner) > 0:
    winner = actual_winner.iloc[0]
    print(f"\n  🏆 Actual winner: {winner[name_col]}")
    print(f"     Our predicted win%: {winner['win']:.2%}")
    print(f"     Our predicted rank: #{int((merged['win'] >= winner['win']).sum())}")
else:
    print(f"\n  ⚠ Could not identify actual winner (finish_pos may not be populated)")

# ── Top 5 comparison ──
actual_top5 = merged[merged["actual_top5"] == 1].sort_values("finish_pos")
print(f"\n  Actual top 5 vs our predictions:")
print(f"  {'Player':<30s} {'Finish':>7s} {'PredWin':>8s} {'PredT5':>8s} {'PredRank':>9s}")
print(f"  {'─' * 65}")
for _, row in actual_top5.iterrows():
    pred_rank = int((merged["win"] >= row["win"]).sum())
    print(f"  {str(row[name_col])[:30]:<30s} "
          f"{row['finish_pos']:>7.0f} "
          f"{row['win']:>7.1%} "
          f"{row['top_5']:>7.1%} "
          f"{'#' + str(pred_rank):>9s}")

# ── Brier scores ──
print(f"\n  ── Scoring Metrics ──")
print(f"  {'Outcome':<12s} {'Brier':>8s} {'BSS':>8s} {'LogLoss':>9s}")
print(f"  {'─' * 40}")

for outcome, pred_col, actual_col in [
    ("Win",      "win",      "actual_win"),
    ("Top 5",    "top_5",    "actual_top5"),
    ("Top 10",   "top_10",   "actual_top10"),
    ("Top 20",   "top_20",   "actual_top20"),
    ("Make Cut", "make_cut", "actual_cut"),
]:
    valid = merged[[pred_col, actual_col]].dropna()
    if len(valid) == 0:
        continue

    bs = brier_score(valid[pred_col], valid[actual_col])
    bss = brier_skill_score(valid[pred_col], valid[actual_col])
    ll = log_loss(valid[pred_col], valid[actual_col])

    print(f"  {outcome:<12s} {bs:>8.5f} {bss:>+7.3f} {ll:>9.4f}")

print(f"\n  Brier Skill Score interpretation:")
print(f"    > 0 = better than naive baseline (predicting base rate for everyone)")
print(f"    ~0.05-0.15 is good for golf (remember: 85% noise)")
print(f"    < 0 = worse than guessing (something is wrong)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: COMPARE TO DATA GOLF (if available)
# ══════════════════════════════════════════════════════════════════════════════
#
# If you have Data Golf's pre-tournament predictions, we can compare
# directly. Their predictions represent the "gold standard" — if our
# model is in the same ballpark, we're doing well.
#
# Note: pre_tournament_pga.json only contains the CURRENT week's
# predictions, not historical ones. So this comparison only works
# if you saved predictions from the week of the target tournament.
# For historical comparison, use the outrights (betting odds) instead.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("STEP 6: Benchmark Comparison")
print("=" * 70)

# ── Try Data Golf pre-tournament predictions ──
dg_path = DATA_DIR / "predictions" / "pre_tournament_pga.json"
if dg_path.exists():
    import json
    with open(dg_path) as f:
        dg_preds = json.load(f)

    # Structure depends on API response format
    if isinstance(dg_preds, list):
        dg_df = pd.DataFrame(dg_preds)
        print(f"  Loaded DG pre-tournament predictions: {len(dg_df)} players")
        print(f"  Columns: {dg_df.columns.tolist()[:10]}")

        # Try to match on dg_id
        if "dg_id" in dg_df.columns:
            dg_compare = merged.merge(
                dg_df[["dg_id", "win_prob"]] if "win_prob" in dg_df.columns
                else dg_df[["dg_id"]],
                on="dg_id", how="inner"
            )
            if "win_prob" in dg_compare.columns:
                print(f"\n  Matched {len(dg_compare)} players with DG predictions")
                print(f"\n  {'Player':<25s} {'Ours':>8s} {'DG':>8s} {'Diff':>8s}")
                print(f"  {'─' * 52}")
                for _, row in dg_compare.nlargest(10, "win").iterrows():
                    diff = row["win"] - row["win_prob"]
                    print(f"  {str(row[name_col])[:25]:<25s} "
                          f"{row['win']:>7.1%} "
                          f"{row['win_prob']:>7.1%} "
                          f"{diff:>+7.1%}")
    else:
        print(f"  DG predictions format: {type(dg_preds).__name__} "
              f"(may need parsing)")
else:
    print(f"  No DG pre-tournament predictions found at {dg_path}")
    print(f"  (This file only captures the current week — that's expected)")

# ── Try historical betting odds as alternative benchmark ──
odds_found = False
for book in ["pinnacle", "draftkings", "fanduel"]:
    odds_path = DATA_DIR / "historical_odds" / f"outrights_win_{book}_{TARGET_YEAR}.json"
    if odds_path.exists():
        import json
        with open(odds_path) as f:
            odds_data = json.load(f)
        print(f"\n  Found {book} odds for {TARGET_YEAR}")
        print(f"  Format: {type(odds_data).__name__}")
        if isinstance(odds_data, list):
            print(f"  Records: {len(odds_data)}")
            if len(odds_data) > 0:
                print(f"  Sample keys: {list(odds_data[0].keys())[:8]}")
        odds_found = True
        break

if not odds_found:
    print(f"\n  No historical odds found for {TARGET_YEAR}")
    print(f"  To enable market comparison, run:")
    print(f"    python data_golf_client.py  (fetches historical odds)")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Tournament:     {TARGET_NAME} ({TARGET_YEAR})")
print(f"  Field size:     {len(field)}")
print(f"  Simulations:    {N_SIMS:,}")

if len(actual_winner) > 0:
    w = actual_winner.iloc[0]
    print(f"  Actual winner:  {w[name_col]}")
    print(f"  Predicted win%: {w['win']:.2%} "
          f"(ranked #{int((merged['win'] >= w['win']).sum())} in our predictions)")

print(f"\n  Next steps:")
print(f"    1. If BSS > 0 for most outcomes → model is adding value")
print(f"    2. If BSS < 0 → check Phase 2 skill estimates are reasonable")
print(f"    3. Run on 5-10 more tournaments to get stable Brier scores")
print(f"    4. Compare against betting odds for a true market benchmark")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from itertools import product
import time as time_module

# Import the core functions from skill_estimation.py
# Adjust this import path to match your project structure
from skill_estimation import (
    compute_reweighted_sg,
    build_all_features,
    FEATURE_COLS,
    TOUR_AVG_SD,
)

DATA_DIR = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# FAST FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
#
# The bottleneck is build_all_features() — it iterates through every player's
# history for each parameter combination. To make the grid search feasible,
# we precompute the raw ingredients once and then quickly recompute weighted
# averages for different parameter values.
# ══════════════════════════════════════════════════════════════════════════════


def precompute_player_histories(df):
    """
    For each player, extract the sorted history arrays needed for
    weighted average computation. This is done ONCE and reused
    across all parameter combinations.
    
    Returns:
        dict: dg_id → {
            'sg_values': array of reweighted SG per round,
            'adj_sg': array of actual adj_sg (the prediction target),
            'days_ago_matrix': for each round i, days since round j (j < i),
            'dates': parsed dates array,
            'indices': original DataFrame indices,
            'calendar_years': array of years,
        }
    """
    print("  Precomputing player histories...")
    
    df = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    
    histories = {}
    
    for pid, group in df.groupby("dg_id"):
        n = len(group)
        
        # Parse dates
        if "start_date" in group.columns:
            dates = pd.to_datetime(group["start_date"], errors="coerce")
        else:
            dates = pd.Series([pd.NaT] * n, index=group.index)
        
        # Precompute days-since matrix for time-weighted averages
        # days_from[i] = array of days from round j to round i, for j < i
        dates_arr = dates.values
        days_from = []
        for i in range(n):
            if pd.isna(dates_arr[i]):
                days_from.append(None)
            else:
                current = dates_arr[i]
                prev_dates = dates_arr[:i]
                valid = ~pd.isna(prev_dates)
                if valid.any():
                    deltas = (current - prev_dates[valid]).astype("timedelta64[D]").astype(float)
                    # Reverse so most recent is first
                    days_from.append(deltas[::-1])
                else:
                    days_from.append(None)
        
        histories[pid] = {
            "indices": group.index.values,
            "adj_sg": group["adj_sg"].values,
            "calendar_years": group["calendar_year"].values,
            "n": n,
            "days_from": days_from,
        }
    
    print(f"  ✓ Precomputed histories for {len(histories):,} players")
    return histories


def fast_compute_features(histories, sg_values_dict, seq_decay, time_half_life,
                          sigma_sq=TOUR_AVG_SD**2):
    """
    Quickly compute features for all player-rounds given precomputed histories.
    
    Instead of the full build_all_features() which re-sorts and re-groups
    the DataFrame each time, this operates directly on the precomputed arrays.
    
    sg_values_dict: dg_id → array of (reweighted) SG values for that player
    """
    all_indices = []
    all_weighted_sg = []
    all_rounds_played = []
    all_days_since = []
    
    for pid, hist in histories.items():
        n = hist["n"]
        sg_vals = sg_values_dict[pid]
        
        weighted_sg = np.full(n, np.nan)
        rounds_played = np.full(n, np.nan)
        days_since = np.full(n, 30.0)  # default
        
        for i in range(n):
            if i == 0:
                rounds_played[i] = 0
                weighted_sg[i] = 0.0
                days_since[i] = 999.0
                continue
            
            rounds_played[i] = i
            
            # Sequence-weighted average (always available)
            hist_sg = sg_vals[:i][::-1]  # most recent first
            seq_w = np.array([seq_decay ** j for j in range(i)])
            seq_w /= seq_w.sum()
            seq_avg = np.dot(seq_w, hist_sg)
            
            # Time-weighted average (if dates available)
            d_from = hist["days_from"][i]
            if d_from is not None and len(d_from) > 0:
                # d_from is already reversed (most recent first)
                # but may be shorter than hist_sg if some dates are missing
                n_timed = len(d_from)
                
                # Get the corresponding SG values (most recent first, same count)
                # We need the last n_timed values before index i, reversed
                timed_sg = sg_vals[i - n_timed:i][::-1]
                
                time_w = 0.5 ** (d_from / time_half_life)
                time_w /= time_w.sum()
                time_avg = np.dot(time_w, timed_sg)
                
                days_since[i] = max(d_from[0], 0)
                
                # Combine via inverse-variance weighting
                seq_var = np.sum(seq_w[:n_timed] ** 2) * sigma_sq
                time_var = np.sum(time_w ** 2) * sigma_sq
                
                if seq_var + time_var > 0:
                    time_frac = (1 / time_var) / (1 / time_var + 1 / seq_var)
                    weighted_sg[i] = time_frac * time_avg + (1 - time_frac) * seq_avg
                else:
                    weighted_sg[i] = seq_avg
            else:
                weighted_sg[i] = seq_avg
        
        all_indices.append(hist["indices"])
        all_weighted_sg.append(weighted_sg)
        all_rounds_played.append(rounds_played)
        all_days_since.append(days_since)
    
    # Concatenate and build feature arrays
    indices = np.concatenate(all_indices)
    weighted_sg = np.concatenate(all_weighted_sg)
    rounds_played = np.concatenate(all_rounds_played)
    days_since = np.clip(np.concatenate(all_days_since), 0, 365)
    
    # Interaction terms
    sg_x_rounds = weighted_sg * rounds_played
    sg_x_days = weighted_sg * days_since
    
    return indices, weighted_sg, rounds_played, days_since, sg_x_rounds, sg_x_days


def evaluate_params(histories, sg_values_dict, adj_sg_array, year_array,
                    seq_decay, time_half_life, test_years=(2023, 2024, 2025)):
    """
    For a given set of (seq_decay, time_half_life), compute features,
    fit the regression on training data, and evaluate walk-forward MSE.
    
    Returns mean MSE across test years.
    """
    # Compute features
    indices, weighted_sg, rounds_played, days_since, sg_x_rounds, sg_x_days = \
        fast_compute_features(histories, sg_values_dict, seq_decay, time_half_life)
    
    # Build feature matrix aligned with original DataFrame order
    n = len(indices)
    X = np.column_stack([weighted_sg, rounds_played, days_since, sg_x_rounds, sg_x_days])
    y = adj_sg_array[indices]
    years = year_array[indices]
    
    # Walk-forward validation
    mses = []
    for test_year in test_years:
        train_mask = (years < test_year) & (rounds_played > 0) & ~np.isnan(weighted_sg)
        test_mask = (years == test_year) & (rounds_played > 0) & ~np.isnan(weighted_sg)
        
        if train_mask.sum() < 100 or test_mask.sum() < 100:
            continue
        
        model = LinearRegression()
        model.fit(X[train_mask], y[train_mask])
        
        preds = model.predict(X[test_mask])
        mse = np.mean((y[test_mask] - preds) ** 2)
        mses.append(mse)
    
    return np.mean(mses) if mses else 999.0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: OPTIMIZE TIME-WEIGHTING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
#
# Grid search over seq_decay and time_half_life.
#
# seq_decay controls how fast the sequence weights fall off:
#   0.93 = aggressive recency (last 20 rounds dominate)
#   0.97 = medium-term (last 50 rounds carry ~70% weight) ← our default
#   0.99 = long-term (last 150 rounds carry ~70% weight)
#
# time_half_life controls the calendar-time decay:
#   60 days = very short-term form emphasis
#   120 days = medium-term ← our default
#   240 days = long-term, slow decay
#
# Data Golf found that the optimal scheme is "medium-term" — recent form
# matters but 2-3 year old data still gets non-zero weight.
# ══════════════════════════════════════════════════════════════════════════════


def optimize_time_weighting(df, histories, sg_weights=None):
    """
    Stage 1: Grid search over seq_decay and time_half_life.
    SG reweighting coefficients are fixed (default or provided).
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Optimizing Time-Weighting Parameters")
    print("=" * 60)
    
    # Reweight SG with current (or default) weights
    if sg_weights is not None:
        df_rw = compute_reweighted_sg(df, sg_weights=sg_weights)
    else:
        df_rw = compute_reweighted_sg(df)
    
    # Build SG values dict for fast computation
    df_sorted = df_rw.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    sg_values_dict = {}
    for pid, group in df_sorted.groupby("dg_id"):
        sg_values_dict[pid] = group["reweighted_sg"].values
    
    # Arrays for evaluation
    adj_sg_array = df_sorted["adj_sg"].values
    year_array = df_sorted["calendar_year"].values
    
    # Define grid
    seq_decays = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    half_lives = [60, 80, 100, 120, 150, 180, 240]
    
    total = len(seq_decays) * len(half_lives)
    print(f"\n  Grid: {len(seq_decays)} decay values × {len(half_lives)} half-lives = {total} combinations")
    print(f"  Evaluating via walk-forward MSE (test years: 2023, 2024, 2025)...\n")
    
    results = []
    best_mse = 999.0
    best_params = {}
    
    for i, (decay, hl) in enumerate(product(seq_decays, half_lives)):
        t0 = time_module.time()
        
        mse = evaluate_params(
            histories, sg_values_dict, adj_sg_array, year_array,
            seq_decay=decay, time_half_life=hl
        )
        
        elapsed = time_module.time() - t0
        
        results.append({
            "seq_decay": decay,
            "time_half_life": hl,
            "mse": mse,
            "rmse": np.sqrt(mse),
        })
        
        if mse < best_mse:
            best_mse = mse
            best_params = {"seq_decay": decay, "time_half_life": hl}
            marker = " ← NEW BEST"
        else:
            marker = ""
        
        if (i + 1) % 7 == 0 or marker:
            print(f"  [{i+1:3d}/{total}] decay={decay:.2f}, hl={hl:3d} → "
                  f"RMSE={np.sqrt(mse):.4f} ({elapsed:.1f}s){marker}")
    
    results_df = pd.DataFrame(results).sort_values("mse")
    
    print(f"\n  ── Stage 1 Results ──")
    print(f"  Best: seq_decay={best_params['seq_decay']}, "
          f"time_half_life={best_params['time_half_life']}")
    print(f"  Best RMSE: {np.sqrt(best_mse):.5f}")
    print(f"\n  Top 10 combinations:")
    print(f"  {'decay':>6s} {'half_life':>10s} {'RMSE':>10s}")
    print(f"  {'─' * 30}")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['seq_decay']:>6.2f} {row['time_half_life']:>10.0f} "
              f"{row['rmse']:>10.5f}")
    
    return best_params, results_df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: OPTIMIZE SG REWEIGHTING COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════
#
# Now fix the time-weighting at Stage 1's optimum and search over the
# SG component weights.
#
# We don't search ALL 4 weights independently — that would be too many
# combinations. Instead we use a constrained search:
#   - β_APP is fixed at 1.0 (the reference point)
#   - β_OTT varies relative to APP (how much MORE predictive is OTT?)
#   - β_ARG varies relative to APP (how much LESS predictive is ARG?)
#   - β_PUTT varies relative to APP (how much LESS predictive is PUTT?)
#
# This is equivalent to asking: "how should we tilt the reweighting
# relative to treating all components equally?"
#
# The Data Golf defaults (1.2, 1.0, 0.9, 0.6) provide our center point.
# We search around these values to see if different weights work better
# on YOUR specific data.
# ══════════════════════════════════════════════════════════════════════════════


def optimize_sg_weights(df, histories, best_time_params):
    """
    Stage 2: Grid search over SG reweighting coefficients.
    Time-weighting params are fixed at Stage 1 optimum.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Optimizing SG Reweighting Coefficients")
    print("=" * 60)
    
    seq_decay = best_time_params["seq_decay"]
    time_half_life = best_time_params["time_half_life"]
    
    print(f"  Fixed params: seq_decay={seq_decay}, time_half_life={time_half_life}")
    
    # Define grid (APP fixed at 1.0 as reference)
    ott_values = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    arg_values = [0.7, 0.8, 0.9, 1.0]
    putt_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    app_fixed = 1.0
    
    total = len(ott_values) * len(arg_values) * len(putt_values)
    print(f"\n  Grid: {len(ott_values)} OTT × {len(arg_values)} ARG × "
          f"{len(putt_values)} PUTT = {total} combinations")
    print(f"  (APP fixed at {app_fixed})\n")
    
    # Precompute sorted DataFrame for reweighting
    df_sorted = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    adj_sg_array = df_sorted["adj_sg"].values
    year_array = df_sorted["calendar_year"].values
    
    results = []
    best_mse = 999.0
    best_weights = {}
    
    for i, (ott, arg, putt) in enumerate(product(ott_values, arg_values, putt_values)):
        sg_weights = {
            "sg_ott": ott,
            "sg_app": app_fixed,
            "sg_arg": arg,
            "sg_putt": putt,
        }
        
        # Reweight SG with these coefficients
        df_rw = compute_reweighted_sg(df_sorted.copy(), sg_weights=sg_weights)
        
        # Build SG values dict
        sg_values_dict = {}
        for pid, group in df_rw.groupby("dg_id"):
            sg_values_dict[pid] = group["reweighted_sg"].values
        
        # Evaluate
        mse = evaluate_params(
            histories, sg_values_dict, adj_sg_array, year_array,
            seq_decay=seq_decay, time_half_life=time_half_life
        )
        
        results.append({
            "sg_ott": ott, "sg_app": app_fixed, "sg_arg": arg, "sg_putt": putt,
            "mse": mse, "rmse": np.sqrt(mse),
        })
        
        if mse < best_mse:
            best_mse = mse
            best_weights = sg_weights.copy()
            marker = " ← NEW BEST"
        else:
            marker = ""
        
        if (i + 1) % 24 == 0 or marker:
            print(f"  [{i+1:3d}/{total}] OTT={ott:.1f} APP={app_fixed:.1f} "
                  f"ARG={arg:.1f} PUTT={putt:.1f} → "
                  f"RMSE={np.sqrt(mse):.5f}{marker}")
    
    results_df = pd.DataFrame(results).sort_values("mse")
    
    print(f"\n  ── Stage 2 Results ──")
    print(f"  Best weights: OTT={best_weights['sg_ott']:.1f}, "
          f"APP={best_weights['sg_app']:.1f}, "
          f"ARG={best_weights['sg_arg']:.1f}, "
          f"PUTT={best_weights['sg_putt']:.1f}")
    print(f"  Best RMSE: {np.sqrt(best_mse):.5f}")
    
    print(f"\n  Top 10 weight combinations:")
    print(f"  {'OTT':>5s} {'APP':>5s} {'ARG':>5s} {'PUTT':>5s} {'RMSE':>10s}")
    print(f"  {'─' * 38}")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['sg_ott']:>5.1f} {row['sg_app']:>5.1f} "
              f"{row['sg_arg']:>5.1f} {row['sg_putt']:>5.1f} "
              f"{row['rmse']:>10.5f}")
    
    # Also show how the default compares
    default_row = results_df[
        (results_df["sg_ott"] == 1.2) & 
        (results_df["sg_arg"] == 0.9) & 
        (results_df["sg_putt"] == 0.6)
    ]
    if not default_row.empty:
        print(f"\n  Default (1.2, 1.0, 0.9, 0.6) RMSE: "
              f"{default_row.iloc[0]['rmse']:.5f}")
    
    return best_weights, results_df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 (OPTIONAL): FINE-GRAINED REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════


def fine_tune(df, histories, coarse_time_params, coarse_sg_weights):
    """
    Stage 3: Fine grid around the Stage 1+2 optimum.
    Searches a narrow range with smaller step sizes.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Fine-Tuning Around Optimum")
    print("=" * 60)
    
    # Fine grid around best seq_decay
    best_decay = coarse_time_params["seq_decay"]
    best_hl = coarse_time_params["time_half_life"]
    
    fine_decays = np.arange(
        max(0.91, best_decay - 0.02), 
        min(0.995, best_decay + 0.025), 
        0.005
    )
    fine_hls = np.arange(
        max(40, best_hl - 30), 
        min(300, best_hl + 40), 
        10
    )
    
    total = len(fine_decays) * len(fine_hls)
    print(f"\n  Fine grid: {len(fine_decays)} decays × {len(fine_hls)} half-lives = {total}")
    print(f"  Center: decay={best_decay}, half_life={best_hl}")
    
    # Reweight with best SG weights
    df_rw = compute_reweighted_sg(
        df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"]).copy(),
        sg_weights=coarse_sg_weights
    )
    
    sg_values_dict = {}
    for pid, group in df_rw.groupby("dg_id"):
        sg_values_dict[pid] = group["reweighted_sg"].values
    
    adj_sg_array = df_rw["adj_sg"].values
    year_array = df_rw["calendar_year"].values
    
    results = []
    best_mse = 999.0
    best_params = {}
    
    for i, (decay, hl) in enumerate(product(fine_decays, fine_hls)):
        mse = evaluate_params(
            histories, sg_values_dict, adj_sg_array, year_array,
            seq_decay=decay, time_half_life=hl
        )
        
        results.append({
            "seq_decay": round(decay, 4),
            "time_half_life": hl,
            "mse": mse,
            "rmse": np.sqrt(mse),
        })
        
        if mse < best_mse:
            best_mse = mse
            best_params = {"seq_decay": round(decay, 4), "time_half_life": hl}
    
    results_df = pd.DataFrame(results).sort_values("mse")
    
    print(f"\n  ── Stage 3 Results ──")
    print(f"  Best: seq_decay={best_params['seq_decay']}, "
          f"time_half_life={best_params['time_half_life']}")
    print(f"  Best RMSE: {np.sqrt(best_mse):.5f}")
    
    print(f"\n  Top 5:")
    for _, row in results_df.head(5).iterrows():
        print(f"    decay={row['seq_decay']:.4f}, hl={row['time_half_life']:.0f} "
              f"→ RMSE={row['rmse']:.5f}")
    
    return best_params, results_df


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON: SHOW IMPROVEMENT OVER BASELINE
# ══════════════════════════════════════════════════════════════════════════════


def compare_baseline_vs_optimized(df, histories, optimized_time, optimized_sg):
    """
    Run the full skill model with default params AND optimized params,
    side by side, to quantify the improvement.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Optimized")
    print("=" * 60)
    
    configs = {
        "Baseline (defaults)": {
            "seq_decay": 0.97,
            "time_half_life": 120,
            "sg_weights": {"sg_ott": 1.2, "sg_app": 1.0, "sg_arg": 0.9, "sg_putt": 0.6},
        },
        "Optimized": {
            "seq_decay": optimized_time["seq_decay"],
            "time_half_life": optimized_time["time_half_life"],
            "sg_weights": optimized_sg,
        },
    }
    
    df_sorted = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    adj_sg_array = df_sorted["adj_sg"].values
    year_array = df_sorted["calendar_year"].values
    
    for name, cfg in configs.items():
        print(f"\n  ── {name} ──")
        print(f"  seq_decay={cfg['seq_decay']}, half_life={cfg['time_half_life']}")
        print(f"  SG weights: OTT={cfg['sg_weights']['sg_ott']}, "
              f"APP={cfg['sg_weights']['sg_app']}, "
              f"ARG={cfg['sg_weights']['sg_arg']}, "
              f"PUTT={cfg['sg_weights']['sg_putt']}")
        
        # Reweight
        df_rw = compute_reweighted_sg(df_sorted.copy(), sg_weights=cfg["sg_weights"])
        sg_values_dict = {}
        for pid, group in df_rw.groupby("dg_id"):
            sg_values_dict[pid] = group["reweighted_sg"].values
        
        # Evaluate per test year
        for test_year in [2023, 2024, 2025]:
            mse = evaluate_params(
                histories, sg_values_dict, adj_sg_array, year_array,
                seq_decay=cfg["seq_decay"],
                time_half_life=cfg["time_half_life"],
                test_years=(test_year,),
            )
            print(f"    {test_year}: RMSE = {np.sqrt(mse):.5f}")
        
        # Overall
        mse_all = evaluate_params(
            histories, sg_values_dict, adj_sg_array, year_array,
            seq_decay=cfg["seq_decay"],
            time_half_life=cfg["time_half_life"],
        )
        print(f"    OVERALL: RMSE = {np.sqrt(mse_all):.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def run_optimization():
    """Full optimization pipeline."""
    
    print("=" * 60)
    print("PHASE 2 OPTIMIZATION: Hyperparameter Tuning")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
    csv_path = DATA_DIR / "processed" / "master_rounds.csv"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Run Phase 1 first")
    
    # Drop rows with missing data
    df = df.dropna(subset=["adj_sg", "round_score", "dg_id"])
    df = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    print(f"  Loaded {len(df):,} player-rounds")
    
    # Precompute histories (done once)
    print("\n2. Precomputing player histories...")
    histories = precompute_player_histories(df)
    
    # Stage 1: Time-weighting optimization
    print("\n3. Stage 1: Time-weighting parameters...")
    best_time, time_results = optimize_time_weighting(df, histories)
    
    # Stage 2: SG reweighting optimization
    print("\n4. Stage 2: SG reweighting coefficients...")
    best_sg, sg_results = optimize_sg_weights(df, histories, best_time)
    
    # Stage 3: Fine-tuning
    print("\n5. Stage 3: Fine-tuning...")
    final_time, fine_results = fine_tune(df, histories, best_time, best_sg)
    
    # Comparison
    compare_baseline_vs_optimized(df, histories, final_time, best_sg)
    
    # Save results
    out_dir = DATA_DIR / "processed"
    time_results.to_csv(out_dir / "optimization_stage1_time_weighting.csv", index=False)
    sg_results.to_csv(out_dir / "optimization_stage2_sg_weights.csv", index=False)
    fine_results.to_csv(out_dir / "optimization_stage3_fine_tune.csv", index=False)
    
    # Save optimal parameters
    optimal_params = {
        "seq_decay": final_time["seq_decay"],
        "time_half_life": final_time["time_half_life"],
        **{f"sg_{k}": v for k, v in best_sg.items()},
    }
    pd.Series(optimal_params).to_json(out_dir / "optimal_params.json")
    
    # Final summary
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Optimal Parameters:")
    print(f"    seq_decay:      {final_time['seq_decay']}")
    print(f"    time_half_life: {final_time['time_half_life']}")
    print(f"    SG weights:     OTT={best_sg['sg_ott']}, APP={best_sg['sg_app']}, "
          f"ARG={best_sg['sg_arg']}, PUTT={best_sg['sg_putt']}")
    print(f"\n  Saved to: {out_dir / 'optimal_params.json'}")
    print(f"\n  Next step: re-run skill_estimation.py with these parameters:")
    print(f"    df, model, diag = run_phase_2(")
    print(f"        seq_decay={final_time['seq_decay']},")
    print(f"        time_half_life={final_time['time_half_life']},")
    print(f"    )")
    print(f"  (and update DEFAULT_SG_WEIGHTS in skill_estimation.py)")
    
    return final_time, best_sg, time_results, sg_results


if __name__ == "__main__":
    final_time, best_sg, time_results, sg_results = run_optimization()
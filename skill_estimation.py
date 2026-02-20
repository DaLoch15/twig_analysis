import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

DATA_DIR = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: SG COMPONENT REWEIGHTING
# ══════════════════════════════════════════════════════════════════════════════
#
# Not all strokes-gained are created equal for PREDICTION. From the Data Golf
# methodology, regressing future total SG on historical SG components gives:
#
#   β_OTT ≈ 1.2   (OTT predicts MORE than 1 stroke of future total SG)
#   β_APP ≈ 1.0   (proportional)
#   β_ARG ≈ 0.9   (slight discount)
#   β_PUTT ≈ 0.6  (heavy discount — putting is mostly noise)
#
# WHY β_OTT > 1.0: This is the cross-category effect. A player who gains
# 1 stroke off the tee also tends to gain ~0.2 strokes on approach in
# future rounds. OTT signals general ball-striking ability — hitting it
# far and straight means you're also hitting good approach shots. So OTT
# is doing double duty as a predictor.
#
# WHAT WE DO: For each round that has SG component data (Shotlink events),
# we compute a "reweighted SG" that amplifies ball-striking and discounts
# putting. For rounds without component data, we use adj_sg as-is.
#
# This means two rounds that were both +2.0 adj_sg get DIFFERENT reweighted
# values if one was driven by OTT (+2.5 reweighted) and the other by putting
# (+1.5 reweighted).
# ══════════════════════════════════════════════════════════════════════════════

# Default coefficients from Data Golf's regression (2) in methodology
DEFAULT_SG_WEIGHTS = {
    "sg_ott": 1.10,
    "sg_app": 1.00,
    "sg_arg": 1.00,
    "sg_putt": 0.80,
}


def compute_reweighted_sg(df, sg_weights=None):
    """
    For each round, compute a reweighted SG that emphasizes ball-striking.
    
    Rounds WITH SG components: reweighted_sg = Σ(β_k × sg_k)
    Rounds WITHOUT SG components: reweighted_sg = adj_sg (unchanged)
    
    Then we normalize so that reweighted_sg has the same mean and variance
    as adj_sg — this ensures the reweighting doesn't shift the overall scale.
    """
    if sg_weights is None:
        sg_weights = DEFAULT_SG_WEIGHTS

    df = df.copy()
    
    has_components = df["has_sg_components"].fillna(False)
    
    # For rounds with SG components, compute reweighted total
    df["reweighted_sg"] = df["adj_sg"].copy()
    
    if has_components.any():
        reweighted = (
            sg_weights["sg_ott"] * df.loc[has_components, "sg_ott"]
            + sg_weights["sg_app"] * df.loc[has_components, "sg_app"]
            + sg_weights["sg_arg"] * df.loc[has_components, "sg_arg"]
            + sg_weights["sg_putt"] * df.loc[has_components, "sg_putt"]
        )
        
        # Normalize reweighted values to match adj_sg scale.
        # Without this, the reweighted values would have different mean/variance
        # than adj_sg, creating a systematic bias between Shotlink and 
        # non-Shotlink rounds.
        adj_sg_mean = df.loc[has_components, "adj_sg"].mean()
        adj_sg_std = df.loc[has_components, "adj_sg"].std()
        rw_mean = reweighted.mean()
        rw_std = reweighted.std()
        
        if rw_std > 0:
            reweighted = (reweighted - rw_mean) / rw_std * adj_sg_std + adj_sg_mean
        
        df.loc[has_components, "reweighted_sg"] = reweighted
    
    n_reweighted = has_components.sum()
    n_total = len(df)
    print(f"  SG reweighting: {n_reweighted:,} / {n_total:,} rounds "
          f"({n_reweighted/n_total:.1%}) had component data")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: DUAL TIME-WEIGHTING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
#
# Data Golf's most clever innovation: instead of one weighting scheme, compute
# TWO weighted averages and combine them.
#
# SEQUENCE-WEIGHTED: Weights decay by ordinal position (most recent round = 1,
#   second most recent = 2, etc.), ignoring calendar gaps. If a player takes
#   6 months off, their last round before the break is still "round 2" in the
#   sequence. This PRESERVES skill history during layoffs.
#
# TIME-WEIGHTED: Weights decay by calendar days since each round. Recent
#   rounds dominate. This captures CURRENT FORM and penalizes stale data.
#
# WHY BOTH? Consider a player returning from a 2-year injury:
#   - Time-weighted: only their 3 recent comeback rounds get weight → high
#     variance, unreliable estimate
#   - Sequence-weighted: their entire career history is smoothly weighted →
#     stable but might miss post-injury changes
#   - Combined: inverse-variance weighting automatically shifts toward the
#     more reliable estimate (sequence-weighted in this case)
#
# For a player on a normal schedule, both estimates are similar and the
# combination offers modest improvement.
# ══════════════════════════════════════════════════════════════════════════════


def sequence_weighted_avg(values, decay=0.97):
    """
    Compute exponentially-weighted average by ordinal position.
    
    values: array sorted most-recent-first
    decay: weight multiplier per position (0.97 = each older round gets
           3% less weight than the one after it)
    
    With decay=0.97:
        - Last 50 rounds carry ~70% of total weight
        - ~150 rounds contribute meaningfully
        - Rounds 200+ have negligible weight
    """
    n = len(values)
    if n == 0:
        return 0.0, np.array([])
    
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()
    avg = np.dot(weights, values)
    return avg, weights


def time_weighted_avg(values, days_ago, half_life=120):
    """
    Compute exponentially-weighted average by calendar time.
    
    values: array of SG values
    days_ago: array of days since each round was played
    half_life: days until weight drops to 50% (120 days ≈ 4 months)
    
    With half_life=120:
        - A round from 4 months ago has 50% weight of today's round
        - A round from 8 months ago has 25% weight
        - A round from 1 year ago has ~12% weight
        - A round from 2 years ago has ~1.5% weight
    """
    n = len(values)
    if n == 0:
        return 0.0, np.array([])
    
    weights = 0.5 ** (np.array(days_ago) / half_life)
    weights /= weights.sum()
    avg = np.dot(weights, values)
    return avg, weights


def combine_estimates(seq_avg, time_avg, seq_weights, time_weights, 
                      sigma_sq=2.9**2):
    """
    Combine sequence-weighted and time-weighted estimates using
    inverse-variance weighting.
    
    The variance of a weighted average = Σ(w_i²) × σ².
    When weights are concentrated on few observations (e.g., time-weighted
    during a long layoff), the variance is high, so that estimate gets
    LESS weight in the combination.
    
    sigma_sq: assumed round-level variance (~2.9² for PGA Tour)
    
    Returns:
        combined estimate, time_fraction (how much weight went to time-weighted)
    """
    if len(seq_weights) == 0:
        return time_avg, 1.0
    if len(time_weights) == 0:
        return seq_avg, 0.0
    
    # Variance of each weighted average
    seq_var = np.sum(seq_weights ** 2) * sigma_sq
    time_var = np.sum(time_weights ** 2) * sigma_sq
    
    # Inverse-variance weighting: lower variance → more weight
    # This is the standard statistical method for combining estimates
    if seq_var + time_var == 0:
        return (seq_avg + time_avg) / 2, 0.5
    
    time_frac = (1 / time_var) / (1 / time_var + 1 / seq_var)
    combined = time_frac * time_avg + (1 - time_frac) * seq_avg
    
    return combined, time_frac


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
#
# For each player at each round, we compute features using ONLY data that
# was available BEFORE that round was played. This is critical — any leakage
# of future data would inflate our metrics and give a false sense of accuracy.
#
# Features computed:
#   1. weighted_sg_avg — combined dual-weighted average of reweighted SG
#   2. rounds_played — cumulative rounds in database
#   3. days_since_last — calendar days since last round
#   4. weighted_sg_avg × rounds_played — interaction for shrinkage
#   5. weighted_sg_avg × days_since_last — interaction for form decay
#
# The interaction terms are where empirical Bayes shrinkage happens
# implicitly through regression. When the regression fits:
#   predicted = β₁×avg + β₄×(avg × rounds)
# it's equivalent to:
#   predicted = (β₁ + β₄×rounds) × avg
# So the effective coefficient on avg INCREASES with more rounds,
# meaning less shrinkage for well-sampled players.
# ══════════════════════════════════════════════════════════════════════════════


def compute_features_for_player(player_df, seq_decay=0.97, time_half_life=120):
    """
    For one player's sorted round history, compute prediction features
    at each time point using only past data.
    
    Returns a DataFrame with the same index as player_df plus feature columns.
    """
    n = len(player_df)
    
    # Pre-allocate feature arrays
    weighted_sg = np.full(n, np.nan)
    rounds_played = np.full(n, np.nan)
    days_since_last = np.full(n, np.nan)
    time_frac = np.full(n, np.nan)
    
    sg_values = player_df["reweighted_sg"].values
    
    # Compute days_ago for time-weighting
    # We need dates — approximate from calendar_year + event_id ordering
    # since we may not have exact dates for all rounds
    if "start_date" in player_df.columns:
        dates = pd.to_datetime(player_df["start_date"], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * n, index=player_df.index)
    
    has_dates = dates.notna()
    
    for i in range(n):
        if i == 0:
            # First round ever — no historical data
            rounds_played[i] = 0
            weighted_sg[i] = 0.0  # will use tour prior
            days_since_last[i] = 999  # large value = "no recent data"
            continue
        
        # Historical data: all rounds BEFORE this one (indices 0 to i-1)
        # Reversed so most recent is first
        hist_sg = sg_values[:i][::-1]
        rounds_played[i] = i
        
        # Sequence-weighted average (always available)
        seq_avg, seq_w = sequence_weighted_avg(hist_sg, decay=seq_decay)
        
        # Time-weighted average (only if we have dates)
        if has_dates.iloc[i] and has_dates.iloc[:i].any():
            current_date = dates.iloc[i]
            hist_dates = dates.iloc[:i]
            valid_time = hist_dates.notna()
            
            if valid_time.any():
                d_ago = (current_date - hist_dates[valid_time]).dt.days.values[::-1]
                hist_sg_timed = sg_values[:i][valid_time.values][::-1]
                
                if len(d_ago) > 0 and d_ago[0] >= 0:
                    time_avg, time_w = time_weighted_avg(hist_sg_timed, d_ago, 
                                                         half_life=time_half_life)
                    combined, tf = combine_estimates(seq_avg, time_avg, 
                                                     seq_w, time_w)
                    weighted_sg[i] = combined
                    time_frac[i] = tf
                    days_since_last[i] = max(d_ago[0], 0)
                else:
                    weighted_sg[i] = seq_avg
                    days_since_last[i] = 30  # default
            else:
                weighted_sg[i] = seq_avg
                days_since_last[i] = 30
        else:
            # No date info — use sequence-weighted only
            weighted_sg[i] = seq_avg
            days_since_last[i] = 30  # reasonable default
    
    # Build feature DataFrame
    features = pd.DataFrame({
        "weighted_sg_avg": weighted_sg,
        "rounds_played": rounds_played,
        "days_since_last": days_since_last,
        "time_frac": time_frac,
    }, index=player_df.index)
    
    return features


def build_all_features(df, seq_decay=0.97, time_half_life=120):
    """
    Compute prediction features for every player-round in the dataset.
    
    This is the most computationally expensive step — iterating through
    each player's history sequentially. For 164k rows across ~2000 players,
    expect 2-5 minutes.
    """
    print(f"  Building features for {df['dg_id'].nunique():,} players...")
    
    # Sort by player then timeline
    df = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])
    
    all_features = []
    player_ids = df["dg_id"].unique()
    
    for i, pid in enumerate(player_ids):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1:,} / {len(player_ids):,} players...")
        
        player_mask = df["dg_id"] == pid
        player_df = df.loc[player_mask]
        features = compute_features_for_player(
            player_df, seq_decay=seq_decay, time_half_life=time_half_life
        )
        all_features.append(features)
    
    features_df = pd.concat(all_features)
    
    # Add interaction terms
    features_df["sg_x_rounds"] = (
        features_df["weighted_sg_avg"] * features_df["rounds_played"]
    )
    features_df["sg_x_days"] = (
        features_df["weighted_sg_avg"] * features_df["days_since_last"]
    )
    
    # Cap days_since_last at 365 to prevent extreme values from
    # dominating the regression
    features_df["days_since_last"] = features_df["days_since_last"].clip(upper=365)
    
    # Merge features back onto main DataFrame
    for col in features_df.columns:
        df[col] = features_df[col]
    
    print(f"  ✓ Features computed for {len(df):,} player-rounds")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: SKILL PREDICTION REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
#
# Now we fit the regression that predicts NEXT round's adj_sg from features
# computed using PAST data only.
#
# This regression IS the skill estimator. Its fitted values are our
# predictions of each player's true skill at each point in time.
#
# The regression implicitly performs empirical Bayes shrinkage through
# the interaction terms:
#
#   predicted = β₁×avg + β₄×(avg × rounds_played)
#            = (β₁ + β₄×rounds_played) × avg
#
# When β₄ > 0 (which it will be), the effective coefficient on avg
# increases with rounds_played. This means:
#   - Player with 10 rounds averaging +3.0 → prediction maybe +0.5
#   - Player with 200 rounds averaging +3.0 → prediction maybe +2.7
#
# The intercept term captures the tour prior: as rounds_played → 0 and
# avg → 0, the prediction converges to the intercept, which will be
# approximately -2.0 (rookies are below-average PGA Tour players).
#
# IMPORTANT: We use walk-forward validation — never train on future data.
# Train on 2017-2022, predict 2023. Train on 2017-2023, predict 2024.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "weighted_sg_avg",
    "rounds_played",
    "days_since_last",
    "sg_x_rounds",
    "sg_x_days",
]


def fit_skill_model(df, train_years=None):
    """
    Fit the skill prediction regression on training data.
    
    Returns:
        model: fitted LinearRegression
        diagnostics: dict with R², coefficients, etc.
    """
    if train_years is None:
        train_years = list(range(2017, 2024))  # train on 2017-2023
    
    # Filter to training period
    train = df[df["calendar_year"].isin(train_years)].copy()
    
    # Drop rows where features or target are missing
    # rounds_played == 0 means no historical data — can't predict
    train = train.dropna(subset=FEATURE_COLS + ["adj_sg"])
    train = train[train["rounds_played"] > 0]
    
    X = train[FEATURE_COLS].values
    y = train["adj_sg"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Diagnostics
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    diagnostics = {
        "r_squared": r_squared,
        "rmse": rmse,
        "n_train": len(train),
        "intercept": model.intercept_,
        "coefficients": dict(zip(FEATURE_COLS, model.coef_)),
    }
    
    return model, diagnostics


def predict_skill(df, model):
    """
    Apply the fitted model to generate skill predictions for all rows.
    
    For rows with no historical data (rounds_played == 0), we assign
    the tour prior (model intercept ≈ -2.0).
    """
    df = df.copy()
    
    # Rows where we can make a prediction
    valid = (
        df[FEATURE_COLS].notna().all(axis=1) 
        & (df["rounds_played"] > 0)
    )
    
    df["predicted_skill"] = np.nan
    
    if valid.any():
        X = df.loc[valid, FEATURE_COLS].values
        df.loc[valid, "predicted_skill"] = model.predict(X)
    
    # For rows with no data, assign the tour prior
    # (approximately the intercept — what the model predicts for a
    # player with 0 weighted average and 0 rounds)
    no_data = df["predicted_skill"].isna()
    df.loc[no_data, "predicted_skill"] = model.intercept_
    
    return df


def print_model_diagnostics(diagnostics):
    """Pretty-print the regression results with interpretation."""
    
    print(f"\n  ── Skill Regression Results ──")
    print(f"  Training samples:  {diagnostics['n_train']:,}")
    print(f"  R² = {diagnostics['r_squared']:.4f}")
    print(f"  RMSE = {diagnostics['rmse']:.3f} strokes")
    print(f"  Intercept = {diagnostics['intercept']:.3f}")
    print(f"    (≈ prediction for unknown player — tour prior)")
    
    print(f"\n  Coefficients:")
    for feat, coef in diagnostics["coefficients"].items():
        print(f"    {feat:<25s} {coef:+.6f}")
    
    # Interpret the shrinkage
    coefs = diagnostics["coefficients"]
    print(f"\n  ── Shrinkage Interpretation ──")
    for n_rounds in [10, 50, 100, 200]:
        # Effective coefficient on weighted_sg_avg at N rounds
        eff_coef = coefs["weighted_sg_avg"] + coefs["sg_x_rounds"] * n_rounds
        print(f"  At {n_rounds:3d} rounds: effective coeff on avg = {eff_coef:.3f} "
              f"(player with +2.0 avg → predicted {diagnostics['intercept'] + eff_coef * 2.0:+.2f})")
    
    # Interpret the rust factor
    print(f"\n  ── Rust Factor Interpretation ──")
    for days in [7, 30, 90, 180, 365]:
        # Impact of days_since_last on prediction
        # Through both direct and interaction effects
        # Assume weighted_sg_avg = +1.0 (solid player)
        direct = coefs["days_since_last"] * days
        interaction = coefs["sg_x_days"] * 1.0 * days
        total = direct + interaction
        print(f"  {days:3d} days off: prediction adjustment = {total:+.3f} strokes "
              f"(for a +1.0 SG player)")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: VARIANCE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
#
# The skill regression gives us the MEAN of each player's scoring distribution.
# We also need the VARIANCE (how much they fluctuate around that mean).
#
# Two players can have identical skill predictions but different variances:
# a consistent player has a narrower distribution (higher win probability
# when they're the best in the field), while a volatile player has a wider
# distribution (lower win probability but higher upset potential).
#
# KEY FINDINGS from the research:
#   1. Estimate variance from RESIDUALS, not raw scores. We want the
#      unpredictable component after accounting for skill changes.
#   2. Player-specific variance is NOT very predictive year-over-year.
#      A high-variance season tends to regress toward average next year.
#      → Shrink player variance estimates toward tour average.
#   3. Course-specific variance IS predictive year-over-year.
#      TPC Sawgrass consistently produces more chaos than Kapalua.
#      → Course variance adjustments are worth modeling (Phase 3).
#   4. Tour average SD ≈ 2.9 strokes per round.
# ══════════════════════════════════════════════════════════════════════════════

TOUR_AVG_SD = 2.9  # strokes per round


def estimate_player_variance(df, min_rounds=30, shrinkage_strength=0.5):
    """
    Estimate player-specific scoring variance from skill regression residuals.
    
    For each player:
        1. Compute residuals: actual adj_sg - predicted_skill
        2. Calculate player's residual SD
        3. Shrink toward tour average (SD ≈ 2.9)
    
    The shrinkage formula:
        weight = n / (n + shrinkage_strength × baseline_n)
        estimated_var = weight × player_var + (1 - weight) × tour_var
    
    This is empirical Bayes for variance estimation. Players with lots of
    data get estimates close to their actual variance. Players with little
    data get estimates close to the tour average.
    """
    df = df.copy()
    tour_var = TOUR_AVG_SD ** 2
    
    # Compute residuals
    df["residual"] = df["adj_sg"] - df["predicted_skill"]
    
    # Player-level variance estimates
    player_stats = df.groupby("dg_id").agg(
        player_var=("residual", "var"),
        n_rounds=("residual", "count"),
    ).reset_index()
    
    # Shrinkage toward tour average
    def shrink_variance(row):
        n = row["n_rounds"]
        pvar = row["player_var"]
        
        if pd.isna(pvar) or n < 5:
            return tour_var  # too few rounds, use tour average
        
        if n < min_rounds:
            # Heavy shrinkage
            baseline_n = min_rounds
        else:
            # Moderate shrinkage even for high-data players
            # (variance is not very predictive year-over-year)
            baseline_n = 20
        
        weight = n / (n + shrinkage_strength * baseline_n)
        return weight * pvar + (1 - weight) * tour_var
    
    player_stats["estimated_var"] = player_stats.apply(shrink_variance, axis=1)
    player_stats["estimated_sd"] = np.sqrt(player_stats["estimated_var"])
    
    # Merge back
    df = df.merge(
        player_stats[["dg_id", "estimated_var", "estimated_sd"]],
        on="dg_id", how="left"
    )
    
    # Fill any remaining NaN with tour average
    df["estimated_var"] = df["estimated_var"].fillna(tour_var)
    df["estimated_sd"] = df["estimated_sd"].fillna(TOUR_AVG_SD)
    
    return df, player_stats


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION: WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Never evaluate a model on data it was trained on. Walk-forward validation
# simulates real-world usage:
#   - Train on 2017-2022, predict 2023
#   - Train on 2017-2023, predict 2024
#   - Train on 2017-2024, predict 2025
#
# Primary metric: Mean Squared Error (MSE) of predicted_skill vs actual adj_sg
# Baseline comparison: MSE of always predicting 0 (tour average)
# ══════════════════════════════════════════════════════════════════════════════


def walk_forward_validation(df, test_years=None):
    """
    Walk-forward validation of the skill model.
    
    For each test year, trains on all prior years and evaluates on the test year.
    Reports MSE and R² for each fold and overall.
    """
    if test_years is None:
        test_years = [2023, 2024, 2025]
    
    results = []
    
    for test_year in test_years:
        train_years = [y for y in range(2017, test_year)]
        
        # Fit on training data
        model, diag = fit_skill_model(df, train_years=train_years)
        
        # Predict on test data
        test = df[df["calendar_year"] == test_year].copy()
        test = test.dropna(subset=FEATURE_COLS + ["adj_sg"])
        test = test[test["rounds_played"] > 0]
        
        if len(test) == 0:
            continue
        
        X_test = test[FEATURE_COLS].values
        y_test = test["adj_sg"].values
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred) ** 2)
        baseline_mse = np.mean(y_test ** 2)  # predicting 0 for everyone
        r2 = 1 - mse / np.var(y_test)
        skill_score = 1 - mse / baseline_mse  # improvement over naive baseline
        
        results.append({
            "test_year": test_year,
            "train_years": f"{min(train_years)}-{max(train_years)}",
            "n_test": len(test),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "baseline_mse": baseline_mse,
            "r2": r2,
            "skill_score": skill_score,
        })
        
        print(f"  {min(train_years)}-{max(train_years)} → {test_year}: "
              f"RMSE={np.sqrt(mse):.3f}, R²={r2:.4f}, "
              f"skill_score={skill_score:.4f} "
              f"(n={len(test):,})")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════


def run_sanity_checks(df, model):
    """Verify the skill estimates make intuitive sense."""
    
    print("\n  ── Sanity Checks ──")
    
    # Check 1: Top predicted players (most recent year with full data)
    latest_year = df["calendar_year"].max()
    latest = df[df["calendar_year"] == latest_year]
    
    # Get each player's average predicted skill in the latest year
    player_skills = latest.groupby(["dg_id", "player_name"]).agg(
        avg_predicted=("predicted_skill", "mean"),
        avg_actual=("adj_sg", "mean"),
        n_rounds=("adj_sg", "count"),
    ).reset_index()
    
    player_skills = player_skills[player_skills["n_rounds"] >= 10]
    
    print(f"\n  Top 15 predicted players ({latest_year}, min 10 rounds):")
    print(f"  {'Player':<30s} {'Predicted':>10s} {'Actual':>8s} {'Rounds':>7s}")
    print(f"  {'─' * 58}")
    
    top = player_skills.nlargest(15, "avg_predicted")
    for _, row in top.iterrows():
        print(f"  {row['player_name']:<30s} "
              f"{row['avg_predicted']:>+10.2f} {row['avg_actual']:>+8.2f} "
              f"{row['n_rounds']:>7.0f}")
    
    # Check 2: Bottom predicted players (should be fringe / low-data)
    print(f"\n  Bottom 10 predicted players ({latest_year}, min 10 rounds):")
    print(f"  {'Player':<30s} {'Predicted':>10s} {'Actual':>8s} {'Rounds':>7s}")
    print(f"  {'─' * 58}")
    
    bottom = player_skills.nsmallest(10, "avg_predicted")
    for _, row in bottom.iterrows():
        print(f"  {row['player_name']:<30s} "
              f"{row['avg_predicted']:>+10.2f} {row['avg_actual']:>+8.2f} "
              f"{row['n_rounds']:>7.0f}")
    
    # Check 3: Predicted vs actual correlation
    corr = player_skills["avg_predicted"].corr(player_skills["avg_actual"])
    print(f"\n  Correlation(predicted, actual) at player level = {corr:.4f}")
    
    # Check 4: Variance estimates
    print(f"\n  Variance estimate distribution:")
    print(f"  {'Stat':<15s} {'Estimated SD':>12s}")
    print(f"  {'─' * 30}")
    for pct in [10, 25, 50, 75, 90]:
        val = df["estimated_sd"].quantile(pct / 100)
        print(f"  {'P' + str(pct):<15s} {val:>12.3f}")
    print(f"  {'Tour avg':<15s} {TOUR_AVG_SD:>12.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_phase_2(seq_decay=0.98, time_half_life=40):
    """
    Full Phase 2 pipeline:
        Load Phase 1 output → reweight SG → build features →
        fit regression → predict skill → estimate variance →
        validate → save
    """
    print("=" * 60)
    print("PHASE 2: Player Skill Estimation")
    print("=" * 60)
    
    # ── Load ──
    print("\n1. Loading Phase 1 output...")
    parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
    csv_path = DATA_DIR / "processed" / "master_rounds.csv"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Run Phase 1 first")
    
    if "adj_sg" not in df.columns:
        raise ValueError("'adj_sg' column missing — run Phase 1 first")
    
    print(f"  Loaded {len(df):,} player-rounds")
    
    # ── Step 1: Reweight SG components ──
    print("\n2. Reweighting SG components...")
    df = compute_reweighted_sg(df)
    
    # ── Step 2: Build features ──
    print("\n3. Building prediction features (this takes a few minutes)...")
    df = build_all_features(df, seq_decay=seq_decay, time_half_life=time_half_life)
    
    # ── Step 3: Fit skill regression ──
    print("\n4. Fitting skill prediction regression...")
    model, diagnostics = fit_skill_model(df, train_years=list(range(2017, 2025)))
    print_model_diagnostics(diagnostics)
    
    # ── Step 4: Predict skill for all rows ──
    print("\n5. Generating skill predictions...")
    df = predict_skill(df, model)
    
    # ── Step 5: Estimate variance ──
    print("\n6. Estimating player-specific variance...")
    for col in ["residual", "estimated_var", "estimated_sd"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df, player_var_stats = estimate_player_variance(df)
    
    # ── Step 6: Walk-forward validation ──
    print("\n7. Walk-forward validation...")
    val_results = walk_forward_validation(df, test_years=[2023, 2024, 2025])
    
    # ── Step 7: Sanity checks ──
    print("\n8. Sanity checks...")
    run_sanity_checks(df, model)
    
    # ── Save ──
    print("\n9. Saving...")
    out_dir = DATA_DIR / "processed"
    
    df.to_parquet(out_dir / "master_rounds.parquet", index=False)
    df.to_csv(out_dir / "master_rounds.csv", index=False)
    print(f"  ✓ Updated master_rounds with skill predictions")
    
    # Save player-level skill summary
    latest = df.groupby(["dg_id", "player_name"]).agg(
        predicted_skill=("predicted_skill", "last"),
        estimated_sd=("estimated_sd", "first"),
        total_rounds=("adj_sg", "count"),
        avg_adj_sg=("adj_sg", "mean"),
    ).reset_index().sort_values("predicted_skill", ascending=False)
    
    latest.to_csv(out_dir / "player_skills.csv", index=False)
    print(f"  ✓ Saved player_skills.csv ({len(latest):,} players)")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  New columns: 'reweighted_sg', 'weighted_sg_avg', 'predicted_skill',")
    print(f"               'estimated_sd', 'estimated_var', 'residual'")
    print(f"  Skill R²:    {diagnostics['r_squared']:.4f}")
    print(f"  Skill RMSE:  {diagnostics['rmse']:.3f} strokes")
    
    return df, model, diagnostics


if __name__ == "__main__":
    df, model, diagnostics = run_phase_2()
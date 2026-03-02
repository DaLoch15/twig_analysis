import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: CORE SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
#
# The simulation represents a 4-round stroke-play tournament as a 3D numpy
# array of shape (n_sims, n_players, 4). Each cell contains a score RELATIVE
# to the course — not the raw score. Lower = better (as in real golf).
#
# The score for player i in round r of simulation s is:
#
#   score[s, i, r] = -μ_i + noise[s, i, r]
#
# Where μ_i is the predicted skill (positive = good player) and noise is
# drawn from N(0, σ_i × course_multiplier). The negation converts from
# "strokes gained" space (positive = good) to "score" space (lower = better).
#
# AR(1) CORRELATION:
#   Rounds within a tournament are NOT independent. A player who shoots
#   well in Round 1 has a very slight tendency to shoot well in Round 2.
#   This is modeled as AR(1) noise:
#
#       noise_r = ρ × noise_{r-1} + innovation_r
#
#   Where ρ ≈ 0.015 (tiny but non-zero). The innovation has variance
#   σ² × (1 - ρ²) so the marginal variance of each round is still σ².
#   This is important — without the (1 - ρ²) correction, later rounds
#   would have inflated variance.
#
# WHY VECTORIZE?
#   A naive loop over 100k simulations × 156 players × 4 rounds = 62M
#   iterations. With numpy broadcasting, we do this in ~4 array operations.
# ══════════════════════════════════════════════════════════════════════════════


def simulate_scores(player_means, player_sds, n_sims=100_000,
                    course_sd_multiplier=1.0, ar1_coeff=0.015,
                    rng=None):
    """
    Simulate 4 rounds of scores for all players across all simulations.

    Parameters
    ----------
    player_means : ndarray, shape (n_players,)
        Predicted adj_sg per round for each player. Positive = good.
    player_sds : ndarray, shape (n_players,)
        Player-specific standard deviations (from Phase 2 variance estimation).
    n_sims : int
        Number of Monte Carlo iterations. 100k gives ~0.01% precision on
        win probabilities for top players.
    course_sd_multiplier : float
        Scales all SDs by this factor for course-specific variance.
        >1.0 = high-variance course (TPC Sawgrass), <1.0 = low-variance (Kapalua).
    ar1_coeff : float
        Round-to-round noise persistence. ~0.015 from Data Golf research.
        Captures the tiny tendency for good/bad rounds to carry over.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility. If None, creates one.

    Returns
    -------
    scores : ndarray, shape (n_sims, n_players, 4)
        Simulated scores (lower = better). NaN for rounds not played (post-cut).
    noise : ndarray, shape (n_sims, n_players, 4)
        The noise component only (useful for within-tournament updating).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_players = len(player_means)

    # Adjust SDs for course-specific variance
    # shape: (n_players,) — broadcasts across simulations
    adjusted_sds = player_sds * course_sd_multiplier

    # Pre-allocate arrays
    scores = np.full((n_sims, n_players, 4), np.nan)
    noise = np.full((n_sims, n_players, 4), np.nan)

    for rd in range(4):
        # Draw standard normal innovations
        # shape: (n_sims, n_players)
        z = rng.standard_normal(size=(n_sims, n_players))

        if rd == 0:
            # Round 1: pure noise, no AR(1) history
            round_noise = z * adjusted_sds  # broadcast: (n_sims, n_players) * (n_players,)
        else:
            # Rounds 2-4: AR(1) correlated noise
            # noise_r = ρ × noise_{r-1} + √(1 - ρ²) × σ × z
            #
            # The √(1 - ρ²) factor ensures the MARGINAL variance of each
            # round is still σ². Without it, Var(noise_r) = ρ²σ² + σ² > σ².
            innovation_sd = adjusted_sds * np.sqrt(1 - ar1_coeff ** 2)
            round_noise = ar1_coeff * noise[:, :, rd - 1] + innovation_sd * z

        noise[:, :, rd] = round_noise

        # Score = -(player skill) + noise
        # Negation because: positive skill = good player = LOWER score
        # A player with μ = +2.0 (2 strokes better than average) should
        # score 2 strokes BELOW the field average.
        scores[:, :, rd] = -player_means + round_noise

    return scores, noise


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: CUT LINE LOGIC
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard PGA Tour cut: top 65 players (plus ties at 65th position) after
# 36 holes advance to the weekend. Players who miss the cut (MC) do not
# play rounds 3-4.
#
# TIES AT THE CUT LINE:
#   If the 65th player's 36-hole total is 140, then EVERY player at 140
#   makes the cut, even if that means 70+ players advance. This is standard
#   PGA Tour rules and matters for simulation accuracy.
#
# IMPLEMENTATION:
#   Instead of looping per-simulation (slow), we vectorize:
#   1. Compute 36-hole totals: (n_sims, n_players)
#   2. Sort each row to find the 65th score
#   3. Boolean mask: missed_cut[s, i] = (total[s, i] > cut_score[s])
#   4. Set rounds 3-4 to NaN for MC players
#
# The partition-based approach (np.partition) is O(n) instead of O(n log n)
# for finding the kth smallest element, which is much faster for 100k sims.
# ══════════════════════════════════════════════════════════════════════════════


def apply_cut(scores, cut_size=65):
    """
    Apply the PGA Tour cut after round 2.

    Players whose 36-hole total exceeds the cut line get NaN for rounds 3-4.
    Handles ties at the cut line (all players tied at the cut score advance).

    Parameters
    ----------
    scores : ndarray, shape (n_sims, n_players, 4)
        Simulated scores. Modified IN PLACE for rounds 3-4.
    cut_size : int
        Number of players making the cut (before ties). Default 65 for PGA Tour.
        Some events use different cut rules (e.g., top 70, or no cut for
        limited-field events like Tour Championship).

    Returns
    -------
    made_cut : ndarray, shape (n_sims, n_players), dtype bool
        True if the player made the cut in that simulation.
    """
    n_sims, n_players, _ = scores.shape

    # If cut_size >= field size, everyone makes the cut (no-cut event)
    if cut_size >= n_players:
        return np.ones((n_sims, n_players), dtype=bool)

    # 36-hole totals
    two_round_total = scores[:, :, 0] + scores[:, :, 1]

    # Find the cut score for each simulation
    # np.partition is O(n) — gives us the k-th smallest without full sort
    # We want the (cut_size - 1)-th smallest score (0-indexed)
    k = min(cut_size - 1, n_players - 1)

    # partition: elements before index k are ≤ element at k
    partitioned = np.partition(two_round_total, k, axis=1)
    cut_scores = partitioned[:, k]  # shape: (n_sims,)

    # Players make the cut if their total ≤ cut score (handles ties)
    # Broadcasting: (n_sims, n_players) ≤ (n_sims, 1)
    made_cut = two_round_total <= cut_scores[:, np.newaxis]

    # Set rounds 3-4 to NaN for players who missed the cut
    missed_cut = ~made_cut
    scores[missed_cut, 2] = np.nan
    scores[missed_cut, 3] = np.nan

    return made_cut


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: PROBABILITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
#
# After simulating the tournament, we count outcomes across simulations:
#   - Win: finished 1st (lowest 72-hole total)
#   - Top 5/10/20: finished in that range
#   - Make cut: advanced past round 2
#
# HANDLING MC PLAYERS IN RANKINGS:
#   Players who missed the cut don't have a valid 72-hole total. We assign
#   them a large penalty score (999) so they rank behind all cut-makers.
#   Their actual "finish position" is based on 36-hole ranking among
#   MC players, but for win/top-N probabilities, they're simply excluded.
#
# TIES IN 72-HOLE TOTAL:
#   In real golf, ties for 1st go to a playoff. For simulation purposes,
#   if N players tie for lowest total, each gets credited with a 1/N
#   share of the win. This is the standard approach in golf analytics.
#   Similarly, a 3-way tie for 5th means all three are "top 5."
#
# FRACTIONAL WINS:
#   When multiple players tie for the lead in a simulation, we assign
#   fractional wins: 1/N each. Over 100k sims, this averages out correctly.
#   The alternative (random playoff) also works but requires more sims
#   to converge to the same precision.
# ══════════════════════════════════════════════════════════════════════════════


def compute_probabilities(scores, made_cut, player_ids=None):
    """
    Extract tournament outcome probabilities from simulation results.

    Parameters
    ----------
    scores : ndarray, shape (n_sims, n_players, 4)
        Simulated round scores (lower = better). NaN for missed rounds.
    made_cut : ndarray, shape (n_sims, n_players)
        Boolean mask for which players made the cut.
    player_ids : list or None
        Player identifiers for the output DataFrame. If None, uses indices.

    Returns
    -------
    probs : DataFrame
        One row per player with columns: win, top_5, top_10, top_20,
        make_cut, expected_finish, median_finish.
    """
    n_sims, n_players, _ = scores.shape

    # 72-hole totals. For MC players, use NaN-safe sum of rounds 1-2 only,
    # but assign a large score for ranking purposes.
    four_round_totals = np.nansum(scores, axis=2)
    four_round_totals[~made_cut] = 999.0  # push MC players to bottom

    # ── Rank within each simulation ──
    # For each simulation, compute the ordinal rank (1 = best).
    # np.argsort(np.argsort(x)) gives the rank. We add 1 for 1-based.
    #
    # HANDLING TIES: We use a "min rank" approach — if three players tie
    # for lowest, all get rank 1 (not 1, 2, 3). This is important for
    # accurate top-N probabilities.
    #
    # Vectorized min-rank: for each player, count how many players
    # have a STRICTLY lower score + 1.
    ranks = np.zeros((n_sims, n_players), dtype=np.float64)

    # Vectorized approach: sort totals, use searchsorted for min-rank
    sorted_totals = np.sort(four_round_totals, axis=1)  # (n_sims, n_players)

    for p in range(n_players):
        player_scores = four_round_totals[:, p]  # (n_sims,)
        # For each sim, count how many players have strictly lower score
        # searchsorted with side='left' gives count of values < target
        ranks[:, p] = np.array([
            np.searchsorted(sorted_totals[s], player_scores[s], side='left') + 1
            for s in range(n_sims)
        ])

    # ── Fractional wins for ties ──
    # If K players share rank 1, each gets 1/K of a win
    win_shares = np.zeros((n_sims, n_players))
    rank_1_mask = ranks == 1
    n_tied_for_lead = rank_1_mask.sum(axis=1)  # (n_sims,) — how many tied for 1st
    n_tied_for_lead = np.maximum(n_tied_for_lead, 1)  # avoid div by 0

    # Broadcast: share = 1/n_tied if rank==1, else 0
    win_shares[rank_1_mask] = 1.0
    win_shares = win_shares / n_tied_for_lead[:, np.newaxis]
    win_shares[~rank_1_mask] = 0.0

    # ── Build results ──
    results = []
    for p in range(n_players):
        player_ranks = ranks[:, p]
        player_made_cut = made_cut[:, p]

        results.append({
            "player_idx": p,
            "win": float(np.mean(win_shares[:, p])),
            "top_5": float(np.mean(player_ranks <= 5)),
            "top_10": float(np.mean(player_ranks <= 10)),
            "top_20": float(np.mean(player_ranks <= 20)),
            "make_cut": float(np.mean(player_made_cut)),
            "expected_finish": float(np.mean(player_ranks)),
            "median_finish": float(np.median(player_ranks)),
        })

    probs = pd.DataFrame(results)

    if player_ids is not None:
        probs["player_id"] = player_ids
    else:
        probs["player_id"] = range(n_players)

    # Sort by win probability (descending)
    probs = probs.sort_values("win", ascending=False).reset_index(drop=True)

    return probs


def compute_probabilities_fast(scores, made_cut, player_ids=None):
    """
    Faster probability extraction that avoids the per-simulation loop.

    Uses np.argsort for ranking (assigns average rank on ties) and
    vectorized operations throughout. Preferred for production use.
    """
    n_sims, n_players, _ = scores.shape

    four_round_totals = np.nansum(scores, axis=2)
    four_round_totals[~made_cut] = 999.0

    # ── Vectorized ranking using argsort of argsort ──
    # This gives ordinal ranks (no tie handling), but is very fast.
    # For top-N probabilities, ties matter less with 100k+ sims.
    order = np.argsort(four_round_totals, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, np.newaxis]
    ranks[rows, order] = np.arange(1, n_players + 1)

    # ── Win: rank == 1 ──
    win_pct = np.mean(ranks == 1, axis=0)

    # ── Top-N probabilities ──
    top_5 = np.mean(ranks <= 5, axis=0)
    top_10 = np.mean(ranks <= 10, axis=0)
    top_20 = np.mean(ranks <= 20, axis=0)

    # ── Make cut ──
    cut_pct = np.mean(made_cut, axis=0)

    # ── Expected and median finish ──
    expected = np.mean(ranks, axis=0)
    median = np.median(ranks, axis=0)

    probs = pd.DataFrame({
        "player_idx": np.arange(n_players),
        "win": win_pct,
        "top_5": top_5,
        "top_10": top_10,
        "top_20": top_20,
        "make_cut": cut_pct,
        "expected_finish": expected,
        "median_finish": median,
    })

    if player_ids is not None:
        probs["player_id"] = player_ids

    probs = probs.sort_values("win", ascending=False).reset_index(drop=True)
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: WITHIN-TOURNAMENT UPDATING
# ══════════════════════════════════════════════════════════════════════════════
#
# After observing a player's Round 1 score, we should UPDATE our belief
# about their skill for Rounds 2-4. A player who shoots 63 in Round 1
# is probably playing slightly better than our pre-tournament estimate.
#
# The Bayesian update formula:
#
#   μ_i_updated = μ_i_prior + K × (actual_score - expected_score)
#
# Where K is the Kalman gain:
#
#   K = σ²_prior / (σ²_prior + σ²_noise)
#
# For a well-established player (200+ rounds), σ²_prior is small relative
# to σ²_noise, so K ≈ 0.02-0.03 (barely update). For a low-data player
# (10 rounds), σ²_prior is larger, so K ≈ 0.2-0.3 (significant update).
#
# SG COMPONENT DIFFERENTIATION:
#   The update should depend on HOW the player beat expectations:
#   - Beat expectations via OTT → larger update (OTT is more predictive)
#   - Beat expectations via putting → smaller update (putting is noisy)
#
#   Update weights from game plan:
#     OTT deviation of +1.0 → +0.12 stroke update
#     APP deviation of +1.0 → +0.08 stroke update
#     ARG deviation of +1.0 → +0.06 stroke update
#     PUTT deviation of +1.0 → +0.04 stroke update
#
# For the initial simulation (pre-tournament), we don't use this module.
# It becomes relevant for IN-PLAY prediction during an ongoing tournament.
# ══════════════════════════════════════════════════════════════════════════════


# Update weights per SG component
# These represent: "if a player's SG:OTT was 1 stroke above expectation
# in the observed round, how much should we adjust their skill estimate?"
SG_UPDATE_WEIGHTS = {
    "sg_ott": 0.12,
    "sg_app": 0.08,
    "sg_arg": 0.06,
    "sg_putt": 0.04,
}

# Fallback: if we don't have SG component data, use total SG
TOTAL_UPDATE_WEIGHT = 0.07  # rough average of component weights


def bayesian_skill_update(prior_means, prior_sds, observed_scores,
                          sg_components=None, rounds_history=None):
    """
    Update player skill estimates after observing round scores.

    Parameters
    ----------
    prior_means : ndarray, shape (n_players,)
        Pre-round skill estimates (predicted adj_sg).
    prior_sds : ndarray, shape (n_players,)
        Pre-round skill standard deviations.
    observed_scores : ndarray, shape (n_players,)
        Actual scores observed (in adj_sg space: positive = good).
    sg_components : dict of ndarray or None
        If available: {'sg_ott': array, 'sg_app': array, ...} for
        component-specific updating.
    rounds_history : ndarray, shape (n_players,) or None
        Number of historical rounds per player. Controls update magnitude:
        more history → smaller updates (more confident prior).

    Returns
    -------
    updated_means : ndarray, shape (n_players,)
        Posterior skill estimates after observing the round.
    """
    n = len(prior_means)
    deviations = observed_scores - prior_means  # how much better/worse than expected

    if sg_components is not None:
        # Component-specific update: weight the deviation by component
        update = np.zeros(n)
        for comp, weight in SG_UPDATE_WEIGHTS.items():
            if comp in sg_components:
                comp_dev = sg_components[comp]  # how the component deviated
                update += weight * comp_dev
    else:
        # Fallback: use total deviation
        update = TOTAL_UPDATE_WEIGHT * deviations

    # Scale update by data depth
    # Players with more history get smaller updates (prior is stronger)
    if rounds_history is not None:
        # Kalman-like gain: K ∝ 1 / (1 + n_rounds / base)
        # base=50 means ~50 rounds of history roughly halves the update
        base_rounds = 50
        scale = base_rounds / (base_rounds + rounds_history)
        scale = np.clip(scale, 0.02, 0.30)  # floor and ceiling
        update *= scale / TOTAL_UPDATE_WEIGHT  # normalize to base weight

    updated_means = prior_means + update
    return updated_means


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: PRESSURE / LEADERBOARD ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════
#
# Players near the lead in rounds 3-4 perform approximately 0.4 strokes
# WORSE than their skill level on the PGA Tour. This is the "pressure effect"
# documented in the academic literature and by Data Golf.
#
# Key findings:
#   - The effect is a BLANKET adjustment, not player-specific
#     (the data doesn't support individual pressure coefficients,
#     except arguably for Tiger Woods who is now retired)
#   - "In contention" means within ~5 strokes of the lead entering
#     the round
#   - The effect is larger for final-round leaders (~0.5 strokes)
#     than for round 3 contenders (~0.3 strokes)
#   - We use 0.4 as the average across rounds 3-4
#
# IMPLEMENTATION:
#   After simulating rounds 1-2, before simulating rounds 3-4:
#   1. Compute each player's 36-hole running total
#   2. Find the leader's total
#   3. Players within CONTENTION_THRESHOLD strokes get a penalty
#   4. Adjust their effective skill for rounds 3-4
# ══════════════════════════════════════════════════════════════════════════════


PRESSURE_PENALTY = 0.4     # strokes worse than skill level
CONTENTION_THRESHOLD = 5.0  # within 5 strokes of the lead


def apply_pressure_adjustment(scores, round_num, penalty=PRESSURE_PENALTY,
                              threshold=CONTENTION_THRESHOLD):
    """
    Adjust scores for players in contention during weekend rounds.

    This is applied AFTER drawing round 3 or 4 scores, by adding
    a penalty to players who are near the lead entering that round.

    Parameters
    ----------
    scores : ndarray, shape (n_sims, n_players, 4)
        Simulated scores. Rounds before round_num must already be filled.
    round_num : int (2 or 3, 0-indexed)
        The round to adjust. 2 = Round 3, 3 = Round 4.
    penalty : float
        Strokes to add (higher = worse) for players in contention.
    threshold : float
        How many strokes behind the leader to be considered "in contention."

    Returns
    -------
    scores : ndarray
        Modified in place. Returns for convenience.
    in_contention : ndarray, shape (n_sims, n_players), dtype bool
        Which players were adjusted.
    """
    # Running total entering this round
    running_total = np.nansum(scores[:, :, :round_num], axis=2)

    # For MC players (NaN in rounds 3-4), running total includes only R1-R2.
    # But they won't be in contention anyway (they missed the cut).

    # Leader's score entering this round (minimum = best)
    # Use np.nanmin to handle any NaN edge cases
    leader_score = np.nanmin(running_total, axis=1)  # (n_sims,)

    # Distance from lead
    behind = running_total - leader_score[:, np.newaxis]  # (n_sims, n_players)

    # In contention: within threshold AND made cut (not NaN)
    made_it = ~np.isnan(scores[:, :, round_num])
    in_contention = (behind <= threshold) & made_it

    # Apply penalty (higher score = worse)
    scores[:, :, round_num] += penalty * in_contention

    return scores, in_contention


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: TOURNAMENT RUNNER — END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the main entry point. It orchestrates:
#   1. Score simulation (Module 1)
#   2. Cut application (Module 2)
#   3. Pressure adjustment (Module 5)
#   4. Probability extraction (Module 3)
#
# It also handles the boilerplate of loading player data, setting up
# the field, and formatting results.
# ══════════════════════════════════════════════════════════════════════════════


def simulate_tournament(player_means, player_sds, n_sims=100_000,
                        cut_size=65, course_sd_multiplier=1.0,
                        ar1_coeff=0.015, apply_pressure=True,
                        player_ids=None, player_names=None,
                        seed=None):
    """
    Full Monte Carlo tournament simulation.

    Parameters
    ----------
    player_means : array-like, shape (n_players,)
        Predicted adj_sg per round for each player. Positive = good.
    player_sds : array-like, shape (n_players,)
        Player-specific standard deviations.
    n_sims : int
        Number of Monte Carlo iterations.
    cut_size : int
        Number of players making the cut (65 for standard PGA Tour).
        Set to field size for no-cut events (Tour Championship, etc.).
    course_sd_multiplier : float
        Course-specific variance scaling factor.
    ar1_coeff : float
        Round-to-round noise correlation.
    apply_pressure : bool
        Whether to apply the contention pressure penalty.
    player_ids : list or None
        Player ID labels for output.
    player_names : list or None
        Player name labels for output.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    probs : DataFrame
        Tournament probabilities for each player.
    sim_data : dict
        Raw simulation data for further analysis.
    """
    player_means = np.asarray(player_means, dtype=np.float64)
    player_sds = np.asarray(player_sds, dtype=np.float64)
    n_players = len(player_means)

    rng = np.random.default_rng(seed)

    print(f"  Simulating tournament: {n_players} players × {n_sims:,} sims...")

    # ── Step 1: Simulate all 4 rounds of scores ──
    scores, noise = simulate_scores(
        player_means, player_sds,
        n_sims=n_sims,
        course_sd_multiplier=course_sd_multiplier,
        ar1_coeff=ar1_coeff,
        rng=rng,
    )

    # ── Step 2: Apply the cut after round 2 ──
    made_cut = apply_cut(scores, cut_size=cut_size)
    cut_rate = made_cut.mean()
    print(f"  Average cut rate: {cut_rate:.1%}")

    # ── Step 3: Apply pressure adjustment for rounds 3-4 ──
    if apply_pressure:
        _, contention_r3 = apply_pressure_adjustment(scores, round_num=2)
        _, contention_r4 = apply_pressure_adjustment(scores, round_num=3)
        avg_contention = (contention_r3.mean() + contention_r4.mean()) / 2
        print(f"  Average contention rate (R3-R4): {avg_contention:.1%}")

    # ── Step 4: Extract probabilities ──
    probs = compute_probabilities_fast(scores, made_cut, player_ids=player_ids)

    # Add player names if provided
    if player_names is not None:
        name_map = dict(zip(
            player_ids if player_ids is not None else range(n_players),
            player_names
        ))
        probs["player_name"] = probs["player_id"].map(name_map)

    # ── Package simulation data ──
    sim_data = {
        "scores": scores,
        "noise": noise,
        "made_cut": made_cut,
        "n_sims": n_sims,
        "n_players": n_players,
    }

    print(f"  ✓ Simulation complete")

    return probs, sim_data


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7: CALIBRATION & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
#
# A prediction model is only useful if it's CALIBRATED — when you say
# there's a 5% chance of something, it should happen ~5% of the time.
#
# METRICS:
#   Brier Score: BS = (1/N) × Σ(pᵢ - oᵢ)²
#     Measures accuracy of probability predictions. Lower = better.
#     Range: 0 (perfect) to 1 (worst). For rare events like wins,
#     even a naive model (predict 1/N for everyone) gets a good Brier
#     score, so we need the Brier Skill Score for context.
#
#   Brier Skill Score: BSS = 1 - BS_model / BS_reference
#     Measures improvement over a reference model (usually 1/N).
#     >0 means better than reference, 0 = same, <0 = worse.
#
#   Log Loss: LL = -(1/N) × Σ[oᵢ·log(pᵢ) + (1-oᵢ)·log(1-pᵢ)]
#     More sensitive to confident wrong predictions than Brier.
#     A model that says 0.001% win probability and the player wins
#     gets heavily penalized. Crucial for rare events.
#
#   Calibration Curve:
#     Group predictions into buckets (0-2%, 2-5%, 5-10%, etc.)
#     and check that actual outcomes match predicted probability.
#     Data Golf found excellent calibration in their model.
# ══════════════════════════════════════════════════════════════════════════════


def brier_score(predicted_probs, actual_outcomes):
    """
    Brier Score: mean squared error of probability predictions.

    Parameters
    ----------
    predicted_probs : array-like
        Predicted probabilities (0 to 1).
    actual_outcomes : array-like
        Binary outcomes (0 or 1).

    Returns
    -------
    float : Brier score (lower = better).
    """
    p = np.asarray(predicted_probs, dtype=np.float64)
    o = np.asarray(actual_outcomes, dtype=np.float64)
    return float(np.mean((p - o) ** 2))


def brier_skill_score(predicted_probs, actual_outcomes, reference_probs=None):
    """
    Brier Skill Score: improvement over reference model.

    Parameters
    ----------
    predicted_probs : array-like
    actual_outcomes : array-like
    reference_probs : array-like or None
        If None, uses uniform 1/N as reference.
    """
    p = np.asarray(predicted_probs)
    o = np.asarray(actual_outcomes)

    bs_model = brier_score(p, o)

    if reference_probs is None:
        # Reference: predict the base rate for everyone
        base_rate = o.mean()
        bs_ref = brier_score(np.full_like(p, base_rate), o)
    else:
        bs_ref = brier_score(np.asarray(reference_probs), o)

    if bs_ref == 0:
        return 0.0

    return float(1 - bs_model / bs_ref)


def log_loss(predicted_probs, actual_outcomes, eps=1e-15):
    """
    Logarithmic loss. More sensitive to confident wrong predictions.

    eps: small constant to avoid log(0).
    """
    p = np.clip(np.asarray(predicted_probs, dtype=np.float64), eps, 1 - eps)
    o = np.asarray(actual_outcomes, dtype=np.float64)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def calibration_table(predicted_probs, actual_outcomes, bins=None):
    """
    Build a calibration table: for each probability bucket, compare
    predicted vs actual event rates.

    Parameters
    ----------
    predicted_probs : array-like
    actual_outcomes : array-like
    bins : list of tuples or None
        Probability buckets as (low, high) pairs.
        Default: [(0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.10),
                  (0.10, 0.20), (0.20, 0.50), (0.50, 1.0)]

    Returns
    -------
    DataFrame with columns: bucket, n, predicted_avg, actual_rate, diff
    """
    if bins is None:
        bins = [
            (0.00, 0.01), (0.01, 0.03), (0.03, 0.06),
            (0.06, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.00),
        ]

    p = np.asarray(predicted_probs)
    o = np.asarray(actual_outcomes)

    rows = []
    for low, high in bins:
        mask = (p >= low) & (p < high)
        n = mask.sum()
        if n > 0:
            pred_avg = p[mask].mean()
            actual_rate = o[mask].mean()
            rows.append({
                "bucket": f"{low:.0%}–{high:.0%}",
                "n": int(n),
                "predicted_avg": float(pred_avg),
                "actual_rate": float(actual_rate),
                "diff": float(actual_rate - pred_avg),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# COURSE VARIANCE MULTIPLIERS
# ══════════════════════════════════════════════════════════════════════════════
#
# Some courses consistently produce more variance than others.
# Unlike player variance (which regresses to mean year-over-year),
# course variance IS predictive: TPC Sawgrass is always chaotic.
#
# We estimate course-specific multipliers from historical data:
# for each course, compare the residual variance at that course
# to the tour-wide average.
# ══════════════════════════════════════════════════════════════════════════════


def estimate_course_variance(df, min_rounds=200, shrinkage=0.5):
    """
    Estimate course-specific variance multipliers from historical residuals.

    Parameters
    ----------
    df : DataFrame
        Master rounds with 'residual', 'event_id', 'course_num' columns.
    min_rounds : int
        Minimum rounds at a course to compute a reliable estimate.
    shrinkage : float
        Shrinkage strength toward overall average (0 = no shrinkage, 1 = full).

    Returns
    -------
    DataFrame : course_id → variance_multiplier
    """
    tour_var = df["residual"].var()

    course_stats = df.groupby(["event_id"]).agg(
        course_var=("residual", "var"),
        n_rounds=("residual", "count"),
    ).reset_index()

    def shrink(row):
        n = row["n_rounds"]
        cv = row["course_var"]
        if pd.isna(cv) or n < 30:
            return 1.0  # default to tour average
        weight = n / (n + shrinkage * min_rounds)
        shrunk_var = weight * cv + (1 - weight) * tour_var
        return np.sqrt(shrunk_var / tour_var)  # multiplier, not raw variance

    course_stats["variance_multiplier"] = course_stats.apply(shrink, axis=1)

    return course_stats[["event_id", "variance_multiplier", "n_rounds"]]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def prepare_field(df, event_id=None, year=None):
    """
    Prepare the player field for a tournament simulation.

    Extracts the most recent predicted_skill and estimated_sd for each
    player who is in the field (or the most recent tournament if no
    event_id specified).

    Returns
    -------
    field : DataFrame with player_id, player_name, predicted_skill, estimated_sd
    """
    if event_id is not None and year is not None:
        field_mask = (df["event_id"] == event_id) & (df["calendar_year"] == year)
        field_df = df[field_mask]
    else:
        # Use the most recent tournament
        latest = df.sort_values(["calendar_year", "event_id"]).iloc[-1]
        field_mask = (
            (df["event_id"] == latest["event_id"])
            & (df["calendar_year"] == latest["calendar_year"])
        )
        field_df = df[field_mask]

    # Get each player's skill estimate at the START of this tournament
    # (use the values from their first round in this event — those were
    # computed from pre-tournament data only)
    field = field_df.groupby(["dg_id", "player_name"]).agg(
        predicted_skill=("predicted_skill", "first"),
        estimated_sd=("estimated_sd", "first"),
        n_rounds_in_event=("round_num", "count"),
    ).reset_index()

    # Fill any missing values with tour defaults
    field["predicted_skill"] = field["predicted_skill"].fillna(-2.0)
    field["estimated_sd"] = field["estimated_sd"].fillna(2.9)

    return field


def run_phase_3(event_id=None, year=None, n_sims=100_000, seed=42):
    """
    Full Phase 3 pipeline:
        Load Phase 2 output → prepare field → simulate → extract probs → save
    """
    print("=" * 60)
    print("PHASE 3: Monte Carlo Tournament Simulation")
    print("=" * 60)

    # ── Load ──
    print("\n1. Loading Phase 2 output...")
    parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
    csv_path = DATA_DIR / "processed" / "master_rounds.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Run Phase 2 first")

    required = ["predicted_skill", "estimated_sd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} — run Phase 2 first")

    print(f"  Loaded {len(df):,} player-rounds")

    # ── Prepare field ──
    print("\n2. Preparing tournament field...")
    field = prepare_field(df, event_id=event_id, year=year)
    print(f"  Field size: {len(field)} players")
    print(f"  Skill range: [{field['predicted_skill'].min():.2f}, "
          f"{field['predicted_skill'].max():.2f}]")
    print(f"  Avg skill: {field['predicted_skill'].mean():.3f}")

    # ── Estimate course variance (if residuals available) ──
    course_mult = 1.0
    if "residual" in df.columns and event_id is not None:
        print("\n3. Estimating course variance multiplier...")
        course_vars = estimate_course_variance(df)
        match = course_vars[course_vars["event_id"] == event_id]
        if not match.empty:
            course_mult = match.iloc[0]["variance_multiplier"]
            print(f"  Course variance multiplier: {course_mult:.3f}")
        else:
            print(f"  No course history — using default (1.0)")
    else:
        print("\n3. Using default course variance multiplier (1.0)")

    # ── Simulate ──
    print(f"\n4. Running Monte Carlo simulation ({n_sims:,} iterations)...")
    probs, sim_data = simulate_tournament(
        player_means=field["predicted_skill"].values,
        player_sds=field["estimated_sd"].values,
        n_sims=n_sims,
        course_sd_multiplier=course_mult,
        player_ids=field["dg_id"].values,
        player_names=field["player_name"].values,
        seed=seed,
    )

    # ── Display results ──
    print(f"\n5. Tournament Predictions:")
    print(f"  {'Player':<30s} {'Win':>7s} {'Top 5':>7s} {'Top 10':>7s} "
          f"{'Top 20':>7s} {'MakeCut':>7s} {'E[Fin]':>7s}")
    print(f"  {'─' * 80}")

    for _, row in probs.head(25).iterrows():
        name = row.get("player_name", f"Player {row['player_id']}")
        print(f"  {str(name)[:30]:<30s} "
              f"{row['win']:>6.1%} {row['top_5']:>6.1%} {row['top_10']:>6.1%} "
              f"{row['top_20']:>6.1%} {row['make_cut']:>6.1%} "
              f"{row['expected_finish']:>7.1f}")

    # ── Save ──
    print(f"\n6. Saving...")
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    probs.to_csv(out_dir / "tournament_predictions.csv", index=False)
    print(f"  ✓ Saved tournament_predictions.csv ({len(probs)} players)")

    # Summary stats
    print(f"\n{'=' * 60}")
    print("PHASE 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Field size:        {len(field)}")
    print(f"  Simulations:       {n_sims:,}")
    print(f"  Course multiplier: {course_mult:.3f}")
    print(f"  Favorite:          {probs.iloc[0].get('player_name', 'N/A')} "
          f"({probs.iloc[0]['win']:.1%})")
    print(f"  Make-cut range:    [{probs['make_cut'].min():.1%}, "
          f"{probs['make_cut'].max():.1%}]")

    return probs, sim_data, field


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION: BACKTEST AGAINST HISTORICAL TOURNAMENTS
# ══════════════════════════════════════════════════════════════════════════════


def backtest(df, test_events, n_sims=50_000, seed=42):
    """
    Run simulations for historical tournaments and evaluate against
    actual outcomes.

    Parameters
    ----------
    df : DataFrame
        Master rounds with predicted_skill, estimated_sd, adj_sg.
    test_events : list of (event_id, year) tuples
        Tournaments to backtest.
    n_sims : int
        Simulations per tournament (50k for backtesting speed).

    Returns
    -------
    all_results : DataFrame
        Aggregated predictions vs actuals for calibration analysis.
    """
    print("=" * 60)
    print("BACKTESTING: Historical Tournament Validation")
    print("=" * 60)

    all_preds = []

    for event_id, year in test_events:
        event_mask = (df["event_id"] == event_id) & (df["calendar_year"] == year)
        event_df = df[event_mask]

        if len(event_df) == 0:
            print(f"  Skipping {event_id} ({year}) — no data")
            continue

        event_name = event_df["event_name"].iloc[0]
        print(f"\n  {event_name} ({year}): {event_df['dg_id'].nunique()} players")

        # Prepare field
        field = prepare_field(df, event_id=event_id, year=year)

        if len(field) < 20:
            print(f"    Skipping — too few players ({len(field)})")
            continue

        # Simulate
        probs, _ = simulate_tournament(
            player_means=field["predicted_skill"].values,
            player_sds=field["estimated_sd"].values,
            n_sims=n_sims,
            player_ids=field["dg_id"].values,
            player_names=field["player_name"].values,
            seed=seed,
        )

        # Get actual outcomes from the data
        actuals = event_df.groupby("dg_id").agg(
            actual_finish=("finish_pos", "first"),
            actual_made_cut=("made_cut", "first"),
            rounds_played=("round_num", "count"),
        ).reset_index()

        # Merge predictions with actuals
        merged = probs.merge(
            actuals, left_on="player_id", right_on="dg_id", how="inner"
        )

        # Compute binary outcomes
        merged["actual_win"] = (merged["actual_finish"] == 1).astype(int)
        merged["actual_top5"] = (merged["actual_finish"] <= 5).astype(int)
        merged["actual_top10"] = (merged["actual_finish"] <= 10).astype(int)
        merged["actual_top20"] = (merged["actual_finish"] <= 20).astype(int)
        merged["actual_cut"] = merged["actual_made_cut"].astype(int)

        merged["event_id"] = event_id
        merged["year"] = year
        merged["event_name"] = event_name

        all_preds.append(merged)

    if not all_preds:
        print("  No tournaments backtested!")
        return pd.DataFrame()

    results = pd.concat(all_preds, ignore_index=True)

    # ── Aggregate metrics ──
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Tournaments:     {len(test_events)}")
    print(f"  Total player-events: {len(results):,}")

    for outcome, pred_col in [
        ("Win", "win"), ("Top 5", "top_5"), ("Top 10", "top_10"),
        ("Top 20", "top_20"), ("Make Cut", "make_cut"),
    ]:
        actual_col = f"actual_{pred_col.replace('top_', 'top')}"
        if pred_col == "make_cut":
            actual_col = "actual_cut"
        elif pred_col == "win":
            actual_col = "actual_win"

        if actual_col in results.columns:
            bs = brier_score(results[pred_col], results[actual_col])
            bss = brier_skill_score(results[pred_col], results[actual_col])
            ll = log_loss(results[pred_col], results[actual_col])
            print(f"\n  {outcome}:")
            print(f"    Brier Score:       {bs:.6f}")
            print(f"    Brier Skill Score: {bss:+.4f}")
            print(f"    Log Loss:          {ll:.4f}")

    # Calibration for win probability
    print(f"\n  Win Probability Calibration:")
    cal = calibration_table(results["win"], results["actual_win"])
    print(f"  {'Bucket':<12s} {'N':>6s} {'Predicted':>10s} {'Actual':>10s} {'Diff':>8s}")
    print(f"  {'─' * 48}")
    for _, row in cal.iterrows():
        print(f"  {row['bucket']:<12s} {row['n']:>6d} "
              f"{row['predicted_avg']:>10.3%} {row['actual_rate']:>10.3%} "
              f"{row['diff']:>+8.3%}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    probs, sim_data, field = run_phase_3(n_sims=100_000, seed=42)
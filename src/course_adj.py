import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from pathlib import Path

DATA_DIR = Path("data")




def build_matrix(df):
    df = df.copy()

    # regress raw score (not sg_total) because sg_total is already determined by round average, need to estimate round avarage after accounting for field strength
    y = df["round_score"].values.astype(float)

    #player identifiers
    player_ids = df["dg_id"].values
    unique_players = np.unique(player_ids)
    player_to_idx = {pid: i for i, pid in enumerate(unique_players)}
    n_players = len(unique_players)

    tourn_rounds = df["tourn_round_id"].values
    unique_tr = np.unique(tourn_rounds)
    tr_to_idx = {tr: i for i, tr in enumerate(unique_tr)}
    n_tourn_rounds = len(unique_tr)

    n_obs = len(df)

   # count rounds per player to determine polynomial degree
    rounds_per_player = df.groupby("dg_id").size().to_dict()

    cols = []
    vals = []
    rows = []

    col_offset = 0
    player_col_ranges = {}  # player_id → (start_col, end_col)

    golf_time = df["golf_time"].values.astype(float)

    # divide by max golf_time keeps polynomial terms in [0, 1] range.
    golf_time_normalized = np.zeros(n_obs)
    for pid in unique_players:
        mask = player_ids == pid
        gt = golf_time[mask]
        max_gt = gt.max()
        if max_gt > 0:
            golf_time_normalized[mask] = gt / max_gt
        else:
            golf_time_normalized[mask] = 0

    for pid in unique_players:
        mask_indices = np.where(player_ids == pid)[0]
        n_rounds = rounds_per_player[pid]
        gt = golf_time_normalized[mask_indices]

        # polynomial degree based on number of rounds:
        if n_rounds >= 200:
            degree = 3 
        elif n_rounds >= 30:
            degree = 2 
        else:
            degree = 1

        start_col = col_offset
        for d in range(1, degree + 1):
            # golf_time^d for this player
            for i, row_idx in enumerate(mask_indices):
                rows.append(row_idx)
                cols.append(col_offset)
                vals.append(gt[i] ** d)
            col_offset += 1

        # layer-specific intercept (1)
        for row_idx in mask_indices:
            rows.append(row_idx)
            cols.append(col_offset)
            vals.append(1.0)
        col_offset += 1

        player_col_ranges[pid] = (start_col, col_offset)

   
    col_offset_tr = col_offset 

    tr_indices = np.array([tr_to_idx[tr] for tr in tourn_rounds])
    for i in range(n_obs):
        rows.append(i)
        cols.append(col_offset_tr + tr_indices[i])
        vals.append(1.0)

    total_cols = col_offset_tr + n_tourn_rounds

    X = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_obs, total_cols)
    )

    feature_info = {
        "n_players": n_players,
        "n_tourn_rounds": n_tourn_rounds,
        "col_offset_tr": col_offset_tr,
        "tr_to_idx": tr_to_idx,
        "unique_tr": unique_tr,
        "player_col_ranges": player_col_ranges,
    }

    return X, y, feature_info



def solve_regression(X, y, feature_info):
    
    #us lsqr to solve least squares regression
    result = lsqr(X, y, damp=0, atol=1e-8, btol=1e-8, show=False)

    beta = result[0]       # coefficient
    istop = result[1]      # convergence flag (1 or 2 = converged)
    n_iters = result[2]    # iterations used
    r_norm = result[3]     # residual norm

    # extract tournament round difficulty 
    col_offset_tr = feature_info["col_offset_tr"]
    unique_tr = feature_info["unique_tr"]
    tr_to_idx = feature_info["tr_to_idx"]

    delta_j = {}
    for tr_id in unique_tr:
        idx = tr_to_idx[tr_id]
        delta_j[tr_id] = beta[col_offset_tr + idx]

    fitted = X.dot(beta)
    residuals = y - fitted

    # check model
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  R² = {r_squared:.4f}")
    print(f"  RMSE = {rmse:.3f} strokes")

    return delta_j, beta, residuals


# compute adjusted strokes-gained for each player-round using the estimated δ_j values:
# adjusted_sg = raw_score - δ_j

def compute_adjusted_sg(df, delta_j):
    df = df.copy()

    df["course_difficulty"] = df["tourn_round_id"].map(delta_j)

    # change δ_j so the mean matches the mean raw score.
    offset = df["round_score"].mean() - df["course_difficulty"].mean()
    df["course_difficulty"] = df["course_difficulty"] + offset

    # Update the delta_j dict too (for the sanity check)
    for key in delta_j:
        delta_j[key] += offset

    # adj sg = -(raw_score - δ_j)
    df["adj_sg"] = -(df["round_score"] - df["course_difficulty"])

    # 0 = average PGA Tour player that year
    yearly_means = df.groupby("calendar_year")["adj_sg"].transform("mean")
    df["adj_sg"] = df["adj_sg"] - yearly_means

    return df


def run_phase_1():
       
       # Load master data → build design matrix → solve regression → compute adjusted SG → sanity check → save

    # load Phase 0 output
    parquet_path = DATA_DIR / "processed" / "master_rounds.parquet"
    csv_path = DATA_DIR / "processed" / "master_rounds.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Run Phase 0 first to create master_rounds")


    # drop rows with missing scores (incomplete rounds, etc.)
    before = len(df)
    df = df.dropna(subset=["round_score", "tourn_round_id", "dg_id", "golf_time"])

    # build design matrix
    X, y, feature_info = build_matrix(df)

    # solve regression
    delta_j, beta, residuals = solve_regression(X, y, feature_info)

    # compute adjusted SG
    df = compute_adjusted_sg(df, delta_j)

    # save results
    out_dir = DATA_DIR / "processed"

    # save updated master with adj_sg column
    df.to_parquet(out_dir / "master_rounds.parquet", index=False)
    df.to_csv(out_dir / "master_rounds.csv", index=False)

    # save δ_j values separately as well
    delta_df = pd.DataFrame([
        {"tourn_round_id": k, "course_difficulty": v}
        for k, v in delta_j.items()
    ])
    delta_df.to_csv(out_dir / "course_difficulty.csv", index=False)

    # Summary
    print(f"  New column: 'adj_sg'")
    print(f"  Mean adj_sg:  {df['adj_sg'].mean():.4f} (should be ~0)")
    print(f"  Std adj_sg:   {df['adj_sg'].std():.3f}")
    print(f"  Range:        [{df['adj_sg'].min():.1f}, {df['adj_sg'].max():.1f}]")

    return df, delta_j


if __name__ == "__main__":
    df, delta_j = run_phase_1()
import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data")


#flatten all into one row per player per round
def load_all_rounds(years=range(2017, 2027)):
    all_rows = []

    for year in years:
        filepath = DATA_DIR / "historical_raw" / f"rounds_pga_{year}.json"
        if not filepath.exists():
            continue

        with open(filepath) as f:
            data = json.load(f)

        year_rows = flatten_year(data, year)
        all_rows.extend(year_rows)

    df = pd.DataFrame(all_rows)
    return df


# change into list of player-round dicts
def flatten_year(data: dict, calendar_year: int) -> list[dict]:

    rows = []

    for event_key, event in data.items():
        #event metadata
        event_meta = {
            "event_id": event.get("event_id"),
            "event_name": event.get("event_name"),
            "season": event.get("season"),
            "event_completed": event.get("event_completed"),
            "calendar_year": calendar_year,
        }

        scores = event.get("scores", [])
        if not isinstance(scores, list):
            continue

        for player in scores:
            # player metadeta
            player_meta = {
                "dg_id": player.get("dg_id"),
                "player_name": player.get("player_name"),
                "fin_text": player.get("fin_text"),
            }

            # get each round
            for round_num in range(1, 5):
                round_key = f"round_{round_num}"
                round_data = player.get(round_key)

    
                if round_data is None or not isinstance(round_data, dict):
                    continue

                #flatten row
                row = {**event_meta, **player_meta, "round_num": round_num}

                for key, value in round_data.items():
                    if key == "score":
                        row["round_score"] = value
                    else:
                        row[key] = value

                rows.append(row)

    return rows


# clean data, type conversions, SG correction
def clean_data(df):
    df = df.copy()


    numeric_cols = [
        "round_score", "course_par",
        "sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g",
        "driving_dist", "driving_acc", "gir", "scrambling",
        "prox_fw", "prox_rgh", "great_shots", "poor_shots",
        "birdies", "pars", "bogies", "doubles_or_worse", "eagles_or_better",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce").astype("Int64")
    df["calendar_year"] = pd.to_numeric(df["calendar_year"], errors="coerce").astype("Int64")

    # score vs par
    if "course_par" in df.columns and "round_score" in df.columns:
        df["score_vs_par"] = df["round_score"] - df["course_par"]

    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    df["has_sg_components"] = df[sg_cols].notna().all(axis=1)

    # fix sg_arg missing value using sg_t2g decomposition
    if all(c in df.columns for c in ["sg_t2g", "sg_ott", "sg_app", "sg_arg"]):
        mask = df["has_sg_components"] & df["sg_t2g"].notna()
        if mask.any():
            sg_arg_recalc = df.loc[mask, "sg_t2g"] - df.loc[mask, "sg_ott"] - df.loc[mask, "sg_app"]
            discrepancy = (df.loc[mask, "sg_arg"] - sg_arg_recalc).abs()
            fix_mask = mask.copy()
            fix_mask.loc[mask] = discrepancy > 0.01
            n_fixes = fix_mask.sum()
            if n_fixes > 0:
                df.loc[fix_mask, "sg_arg"] = sg_arg_recalc[fix_mask]


    if df["has_sg_components"].any():
        mask = df["has_sg_components"]
        sg_sum = df.loc[mask, sg_cols].sum(axis=1)
        sg_diff = (sg_sum - df.loc[mask, "sg_total"]).abs()
        n_mismatches = (sg_diff > 0.1).sum()
        pct_clean = (1 - n_mismatches / mask.sum()) * 100

    # made cut
    if "fin_text" in df.columns:
        df["made_cut"] = ~df["fin_text"].astype(str).str.upper().str.contains("MC|CUT|WD|DQ", na=False)
    else:
        df["made_cut"] = df.groupby(
            ["dg_id", "event_id", "calendar_year"]
        )["round_num"].transform("max") > 2

    # finish position
    if "fin_text" in df.columns:
        df["finish_pos"] = (
            df["fin_text"]
            .astype(str)
            .str.replace("T", "", regex=False)
            .str.strip()
        )
        df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")

    df["tourn_round_id"] = (
        df["event_id"].astype(str) + "_"
        + df["calendar_year"].astype(str) + "_"
        + df["course_num"].astype(str) + "_"
        + df["round_num"].astype(str)
    )

    # field size
    df["field_size"] = df.groupby(
        ["event_id", "calendar_year"]
    )["dg_id"].transform("nunique")

    return df


#add player details
def add_player_metadata(df):

    player_path = DATA_DIR / "general" / "player_list.json"
    if not player_path.exists():
        return df

    with open(player_path) as f:
        players = pd.DataFrame(json.load(f))

    merge_cols = [c for c in ["dg_id", "country", "amateur"] if c in players.columns]
    if "dg_id" in merge_cols and len(merge_cols) > 1:
        df = df.merge(players[merge_cols], on="dg_id", how="left")

    return df

#add tournament details
def add_tournament_metadata(df):
    frames = []
    for year in range(2017, 2027):
        path = DATA_DIR / "general" / f"schedule_{year}.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
                # Schedule may be a dict with a 'schedule' key or a flat list
                if isinstance(raw, dict) and "schedule" in raw:
                    frames.append(pd.DataFrame(raw["schedule"]))
                elif isinstance(raw, list):
                    frames.append(pd.DataFrame(raw))

    if not frames:
        return df

    schedules = pd.concat(frames, ignore_index=True)

    join_keys = [c for c in ["event_id", "calendar_year", "season"] 
                 if c in schedules.columns and c in df.columns]
    info_cols = [c for c in ["course_latitude", "course_longitude", 
                              "latitude", "longitude", "city", "country",
                              "start_date", "end_date"]
                 if c in schedules.columns]

    if join_keys and info_cols:
        # avoid column conflicts
        info_cols_safe = [c for c in info_cols if c not in df.columns or c in join_keys]
        merge_subset = schedules[join_keys + info_cols_safe].drop_duplicates(subset=join_keys)
        df = df.merge(merge_subset, on=join_keys, how="left")

    return df


def compute_derived_features(df):
    df = df.copy()

    df = df.sort_values(["dg_id", "calendar_year", "event_id", "round_num"])

   #calculate golf time
    df["golf_time"] = df.groupby("dg_id").cumcount() + 1

    # total round per player
    df["player_total_rounds"] = df.groupby("dg_id")["dg_id"].transform("count")

    # field avg score
    df["round_field_avg"] = df.groupby("tourn_round_id")["round_score"].transform("mean")
    df["round_field_size"] = df.groupby("tourn_round_id")["dg_id"].transform("count")

    # player number of tourneys
    df["rounds_in_event"] = df.groupby(
        ["dg_id", "event_id", "calendar_year"]
    )["round_num"].transform("max")

    return df


#bring it all together

def build_master_dataset():
   
    df = load_all_rounds()

    df = clean_data(df)

    df = add_player_metadata(df)

    df = add_tournament_metadata(df)

    df = compute_derived_features(df)

    # save
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "master_rounds.parquet"
    df.to_parquet(parquet_path, index=False)

    csv_path = out_dir / "master_rounds.csv"
    df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    df = build_master_dataset()
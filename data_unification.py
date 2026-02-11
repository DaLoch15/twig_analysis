import pandas as pd
import numpy as np
import json
from pathlib import Path


DATA_DIR = Path("data")


#concatenate all rounds data into one dataframe, each row a player-round combination

def load_all_rounds(years=range(2017, 2027)):


    frames = []

    for year in years:
        file_path = DATA_DIR / "historical_raw" / f"rounds_pga_{year}.json"

        if not filepath.exists():
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        frames.append(df)

    master_df = pd.concat(frames, ignore_index=True)

    return master_df




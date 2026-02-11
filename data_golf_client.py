import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
BASE_URL = "https://feeds.datagolf.com"
DATA_DIR = Path("data")

# fetch and save data, general function

def fetch_endpoint(endpoint: str, params: dict = None, label: str = "") -> dict | list | None:
    if params is None:
        params = {}
    params["key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint}: {e}")
        return None


def save_data(data, category: str, filename: str):

    out_dir = DATA_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    #save JSON
    json_path = out_dir / f"{filename}.json"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Saved JSON to {json_path}")
    
    #save CSV if data is a list of dicts
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
        csv_path = out_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")


def fetch_and_save(endpoint, params, category, filename, label = ""):
    data = fetch_endpoint(endpoint, params, label or filename)
    if data is not None:
        save_data(data, category, filename)
    
    time.sleep(1)

    return data


#start fetching the general categories using helper functions

def fetch_general_use():

    fetch_and_save(
        "get-player-list", {},
        "general", "player_list",
        "Player List & IDs"
    )

    #tour schedules
    for season in range(2024, 2027):
        fetch_and_save(
            "get-schedule", {"tour": "pga", "season": season},
            "general", f"schedule_{season}",
            f"Schedule PGA {season}"
        )

    #field updates
    fetch_and_save(
        "field-updates", {"tour": "pga"},
        "general", "field_updates_pga",
        "Field Updates: PGA"
    )

  #DG model predictions
def fetch_model_predictions():

    fetch_and_save(
        "preds/get-dg-rankings", {},
        "predictions", "dg_rankings",
        "DG Rankings"
    )

    #player skill ratings
    fetch_and_save(
        "preds/skill-ratings", {"display": "value"},
        "predictions", "skill_ratings_value",
        "Skill Ratings (values)"
    )
    fetch_and_save(
        "preds/skill-ratings", {"display": "rank"},
        "predictions", "skill_ratings_rank",
        "Skill Ratings (ranks)"
    )

    fetch_and_save(
        "preds/player-decompositions", {"tour": "pga"},
        "predictions", "player_decompositions_pga",
        "Player Decompositions PGA"
    )

    fetch_and_save(
        "preds/pre-tournament",
        {"tour": "pga", "odds_format": "percent"},
        "predictions", "pre_tournament_pga",
        "Pre-Tournament Predictions PGA"
    )

    # appraoch skill predictions
    for period in ["l24", "l12"]:
        fetch_and_save(
            "preds/approach-skill", {"period": period},
            "predictions", f"approach_skill_{period}",
            f"Approach Skill ({period})"
        )  

#Historical Raw Data
def fetch_historical_raw_data(years=None):
    
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

    fetch_and_save(
            "historical-raw-data/event-list", {"tour": "pga"},
            "historical_raw", f"event_list_pga",
            f"Raw Data Event List: PGA"
        )
    
    for year in years:
        fetch_and_save(
            "historical-raw-data/rounds",
            {"tour": "pga", "event_id": "all", "year": str(year)},
            "historical_raw", f"rounds_pga_{year}",
            f"Round Data: PGA {year}"
            )
    time.sleep(1)    

# get historical_event_stat

def fetch_historical_event_stats():

    event_list = fetch_and_save(
        "historical-event-data/event-list", {"tour": "pga"},
        "historical_events", "event_list_pga",
        "Event Stats Event List"
    )

    if event_list:
        for year in ["2025", "2026"]:
            for event in event_list[:5] if isinstance(event_list, list) else []:
                eid = event.get("event_id") or event.get("id")
                if eid:
                    fetch_and_save(
                        "historical-event-data/events",
                        {"tour": "pga", "event_id": str(eid), "year": year},
                        "historical_events", f"event_{eid}_{year}",
                        f"Event {eid} ({year})"
                    )


#odds data, best for comparing models predictions against market-implied probabilities
def fetch_historical_odds(years=None):
    if years is None:
        years = [2023, 2024, 2025]

    
    fetch_and_save(
        "historical-odds/event-list", {"tour": "pga"},
        "historical_odds", "event_list_pga",
        "Odds Event List PGA"
    )

    for year in years:
        for book in ["pinnacle", "draftkings", "fanduel"]:
            fetch_and_save(
                "historical-odds/outrights",
                {"tour": "pga", "event_id": "all", "year": str(year),
                 "market": "win", "book": book, "odds_format": "decimal"},
                "historical_odds", f"outrights_win_{book}_{year}",
                f"Outrights Win {book} {year}"
            )



if __name__ == "__main__":
    start_time = time.time()
    print(f"Data fetching started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    fetch_general_use()
    fetch_model_predictions()
    fetch_historical_raw_data()
    fetch_historical_event_stats()
    fetch_historical_odds()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data fetching completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")   





    





import json
from pathlib import Path

p = Path('data/historical_raw/rounds_pga_2024.json')
with open(p) as f:
    data = json.load(f)

first_key = list(data.keys())[0]
event = data[first_key]
scores = event['scores']
print(f'scores type: {type(scores).__name__}')
if isinstance(scores, list) and len(scores) > 0:
    print(f'scores length: {len(scores)}')
    print(f'rows: {list(scores[0].keys())}')
    print(json.dumps(scores[0], indent=2))
elif isinstance(scores, dict):
    first_score_key = list(scores.keys())[0]
    print(f'scores keys (first 5): {list(scores.keys())[:5]}')
    print(json.dumps(scores[first_score_key], indent=2) if isinstance(scores[first_score_key], dict) else scores[first_score_key])
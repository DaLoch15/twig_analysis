import json
from pathlib import Path

df
scheffler = df[df["player_name"].str.contains("Scheffler", na=False)]
print(scheffler.groupby("calendar_year")["predicted_skill"].mean())
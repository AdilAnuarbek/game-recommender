import requests
import time
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAWG_API_KEY")
BASE_URL = "https://api.rawg.io/api/games"
INPUT_PATH = "data/processed/games.parquet"
OUTPUT_PATH = "data/processed/games_enriched.parquet"

TOP_N = 5000  # enrich only top N games by ratings_count to save API calls


def fetch_description(game_id: int) -> str:
    try:
        response = requests.get(f"{BASE_URL}/{game_id}", params={"key": API_KEY})
        response.raise_for_status()
        data = response.json()
        return data.get("description_raw", "").strip()[:500]
    except Exception as e:
        print(f"Failed for game_id {game_id}: {e}")
        return ""


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    top = df.nlargest(TOP_N, "ratings_count").copy()
    rest = df[~df["id"].isin(top["id"])].copy()

    # Skip games that already have a description
    already_done = top[top["description"].str.strip().str.len() > 0]
    to_fetch = top[top["description"].str.strip().str.len() == 0].copy()
    print(f"Skipping {len(already_done)} already enriched. Fetching {len(to_fetch)} remaining...")

    descriptions = {}
    for game_id in tqdm(to_fetch["id"], desc="Fetching descriptions"):
        descriptions[game_id] = fetch_description(game_id)
        time.sleep(0.25)

    top.loc[to_fetch["id"].index, "description"] = to_fetch["id"].map(descriptions)

    # Rebuild embedding_text for all
    top["embedding_text"] = (
        top["name"] + ". "
        + "Genres: " + top["genres"] + ". "
        + "Tags: " + top["tags"] + ". "
        + top["description"].fillna("")
    ).str.strip(". ")

    return pd.concat([top, rest], ignore_index=True)


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} games. Enriching top {TOP_N}...")
    df = enrich(df)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved enriched data to {OUTPUT_PATH}")
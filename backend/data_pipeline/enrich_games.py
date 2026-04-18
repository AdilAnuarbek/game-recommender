import requests
import time
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

API_KEY = os.getenv("RAWG_API_KEY")
BASE_URL = "https://api.rawg.io/api/games"
INPUT_PATH = "data/processed/games.parquet"
OUTPUT_PATH = "data/processed/games_enriched.parquet"
MAX_WORKERS = 10  # 10 concurrent requests — safe for RAWG free tier


def fetch_description(game_id: int) -> tuple[int, str]:
    try:
        response = requests.get(
            f"{BASE_URL}/{game_id}",
            params={"key": API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        desc = response.json().get("description_raw", "").strip()[:500]
        return game_id, desc
    except Exception as e:
        print(f"Failed for game_id {game_id}: {e}")
        return game_id, ""


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    to_fetch = df[df["description"].str.strip().str.len() == 0].copy()
    already_done = df[df["description"].str.strip().str.len() > 0].copy()
    print(f"Skipping {len(already_done)} already enriched. Fetching {len(to_fetch)} remaining...")

    descriptions = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_description, gid): gid for gid in to_fetch["id"]}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching descriptions"):
            game_id, desc = future.result()
            descriptions[game_id] = desc

    df.loc[to_fetch["id"].index, "description"] = to_fetch["id"].map(descriptions)

    df["embedding_text"] = (
        df["name"] + ". "
        + "Genres: " + df["genres"] + ". "
        + "Tags: " + df["tags"] + ". "
        + df["description"].fillna("")
    ).str.strip(". ")

    return df


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} games. Enriching all games...")
    df = enrich(df)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved enriched data to {OUTPUT_PATH}")
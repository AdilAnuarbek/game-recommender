import requests
import json
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("RAWG_API_KEY")
BASE_URL = "https://api.rawg.io/api/games"
OUTPUT_PATH = "data/raw/games.json"
MAX_PAGES = 125  # 125 pages × 40 results = ~5,000 games to start

# Fetches the data of the page {page} and returns json as a list
def fetch_page(page: int) -> dict:
    params = {
        "key": API_KEY,
        "page": page,
        "page_size": 40,
        "ordering": "-rating",
        "min_playtime": 1,
        "ratings_count": 20,  # skip obscure games with almost no ratings
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


# Fetches data for all pages until {MAX_PAGES} and returns it as a list
def fetch_all_games() -> list:
    all_games = []

    for page in tqdm(range(1, MAX_PAGES + 1), desc="Fetching games"):
        try:
            data = fetch_page(page)
            games = data.get("results", [])
            if not games:
                print(f"No results on page {page}, stopping.")
                break
            all_games.extend(games)
            time.sleep(0.25)  # be respectful to the API, avoid rate limiting
        except requests.HTTPError as e:
            print(f"HTTP error on page {page}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error on page {page}: {e}")
            break

    return all_games


# Saves the fetched data {games} into a json file {OUTPUT_PATH}
def save_games(games: list):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(games)} games to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Starting RAWG data fetch...")
    games = fetch_all_games()
    save_games(games)
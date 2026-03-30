import json
import os
import pandas as pd

INPUT_PATH = "data/raw/games.json"
OUTPUT_PATH = "data/processed/games.parquet"

# Returns the raw data from the {path} json file
def load_raw(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fields(game: dict) -> dict:
    return {
        "id":              game.get("id"),
        "name":            game.get("name", "").strip(),
        "released":        game.get("released"),
        "rating":          game.get("rating"),
        "ratings_count":   game.get("ratings_count"),
        "playtime":        game.get("playtime"),
        "description":     "",
        "genres":          ", ".join(g["name"] for g in game.get("genres", [])),
        "tags":            ", ".join(t["name"] for t in game.get("tags", [])[:10]),  # top 10 tags only
        "platforms":       ", ".join(
                               p["platform"]["name"] for p in game.get("platforms", [])
                           ),
        "background_image": game.get("background_image", ""),
        "slug":            game.get("slug", ""),
    }


def clean(games: list) -> pd.DataFrame:
    records = [extract_fields(g) for g in games]
    df = pd.DataFrame(records)

    # Drop rows missing critical fields
    df.dropna(subset=["id", "name", "rating"], inplace=True)

    # Drop games with suspiciously low engagement
    df = df[df["ratings_count"] >= 20]

    # Drop duplicate game IDs
    df.drop_duplicates(subset="id", inplace=True)

    # Normalize rating to 0–1 scale (RAWG uses 0–5)
    df["rating_normalized"] = df["rating"] / 5.0

    # Fill missing playtime with genre-group median
    df["playtime"] = df.groupby("genres")["playtime"].transform(
        lambda x: x.fillna(x.median())
    )
    df["playtime"].fillna(df["playtime"].median(), inplace=True)  # fallback global median

    # Build the text representation used for embedding later
    df["embedding_text"] = (
        df["name"] + ". "
        + "Genres: " + df["genres"] + ". "
        + "Tags: " + df["tags"] + ". "
        + df["description"].fillna("")
    ).str.strip(". ")

    # Reset index cleanly
    df.reset_index(drop=True, inplace=True)

    return df


def save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} cleaned games to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Loading raw data...")
    raw = load_raw(INPUT_PATH)
    print(f"Loaded {len(raw)} raw records.")

    print("Cleaning...")
    df = clean(raw)
    print(f"{len(df)} games after cleaning.")
    print(df[["name", "rating", "genres", "tags"]].head(10))

    save(df)
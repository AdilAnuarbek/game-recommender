from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "games"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

model = TextEmbedding(model_name=EMBEDDING_MODEL)
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def build_query_text(query: str, liked_games: list[str]) -> str:
    parts = []
    if query:
        parts.append(query)
    if liked_games:
        parts.append("Similar to: " + ", ".join(liked_games))
    return ". ".join(parts)


def search_games(query: str, liked_games: list[str], top_k: int = 20) -> list[dict]:
    query_text = build_query_text(query, liked_games)
    vector = list(model.embed([query_text]))[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="rating",
                    range=Range(gte=3.5)
                )
            ]
        )
    )

    candidates = []
    for r in results.points:
        p = r.payload
        candidates.append({
            "name":             p.get("name", ""),
            "genres":           p.get("genres", ""),
            "tags":             p.get("tags", ""),
            "rating":           p.get("rating", 0.0),
            "ratings_count":    p.get("ratings_count", 0),
            "playtime":         p.get("playtime", 0.0),
            "released":         p.get("released", ""),
            "background_image": p.get("background_image", ""),
            "slug":             p.get("slug", ""),
            "score":            r.score,
        })

    # Blend vector similarity with popularity
    # Normalize ratings_count to 0-1 using log scale (avoids Minecraft dominating everything)
    import math
    max_log = max(math.log1p(c["ratings_count"]) for c in candidates) or 1
    for c in candidates:
        popularity = math.log1p(c["ratings_count"]) / max_log
        c["blended_score"] = 0.7 * c["score"] + 0.3 * popularity

    # Sort by blended score so LLM sees best candidates first
    candidates.sort(key=lambda x: x["blended_score"], reverse=True)

    return candidates
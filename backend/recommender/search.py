from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "games"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load once at module level — avoids reloading on every request
model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host="localhost", port=6333)


def build_query_text(query: str, liked_games: list[str]) -> str:
    parts = []
    if query:
        parts.append(query)
    if liked_games:
        parts.append("Similar to: " + ", ".join(liked_games))
    return ". ".join(parts)


def search_games(query: str, liked_games: list[str], top_k: int = 20) -> list[dict]:
    query_text = build_query_text(query, liked_games)
    vector = model.encode([query_text], normalize_embeddings=True)[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    )

    candidates = []
    for r in results.points:
        p = r.payload
        candidates.append({
            "name":             p.get("name", ""),
            "genres":           p.get("genres", ""),
            "tags":             p.get("tags", ""),
            "rating":           p.get("rating", 0.0),
            "playtime":         p.get("playtime", 0.0),
            "released":         p.get("released", ""),
            "background_image": p.get("background_image", ""),
            "score":            r.score,
        })

    return candidates
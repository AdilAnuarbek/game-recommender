import pandas as pd
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = "data/processed/games_enriched.parquet"
COLLECTION_NAME = "games"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df = df[df["embedding_text"].str.strip().str.len() > 20].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} games to embed.")
    return df


def setup_qdrant(client: QdrantClient, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection '{COLLECTION_NAME}'.")


def embed_and_upload(df: pd.DataFrame, model: TextEmbedding, client: QdrantClient):
    texts = df["embedding_text"].tolist()

    print("Generating embeddings...")
    embeddings = list(tqdm(model.embed(texts), total=len(texts), desc="Embedding games"))

    print("Uploading to Qdrant...")
    points = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Building points")):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row["id"]))),
                vector=embeddings[i].tolist(),
                payload={
                    "game_id":          int(row["id"]),
                    "name":             row["name"],
                    "genres":           row["genres"],
                    "tags":             row["tags"],
                    "rating":           float(row["rating"]),
                    "ratings_count":    int(row["ratings_count"]),
                    "playtime":         float(row["playtime"]) if pd.notna(row["playtime"]) else 0.0,
                    "released":         str(row["released"]) if pd.notna(row["released"]) else "",
                    "background_image": row["background_image"],
                    "slug":             row["slug"],
                    "embedding_text":   row["embedding_text"],
                },
            )
        )

    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + BATCH_SIZE],
        )

    print(f"Successfully uploaded {len(points)} game vectors to Qdrant.")


if __name__ == "__main__":
    df = load_data()

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # FastEmbed doesn't have get_sentence_embedding_dimension() — hardcode 384 for all-MiniLM-L6-v2
    setup_qdrant(client, vector_size=384)
    embed_and_upload(df, model, client)

    # Sanity check
    test_text = "An open world fantasy RPG with exploration and combat"
    test_vector = list(model.embed([test_text]))[0].tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=test_vector, limit=5)
    print("\nSanity check — top 5 results for 'open world fantasy RPG':")
    for r in results.points:
        print(f"  {r.payload['name']} (score: {r.score:.3f})")
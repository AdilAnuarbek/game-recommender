import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import uuid

INPUT_PATH = "data/processed/games_enriched.parquet"
COLLECTION_NAME = "games"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, 384-dim vectors
BATCH_SIZE = 128


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    # Only embed games that have meaningful embedding_text
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


def embed_and_upload(df: pd.DataFrame, model: SentenceTransformer, client: QdrantClient):
    texts = df["embedding_text"].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity works best with normalized vectors
    )

    print("Uploading to Qdrant...")
    points = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Building points")):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row["id"]))),  # stable UUID from game id
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

    # Upload in batches to avoid memory issues
    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + BATCH_SIZE],
        )

    print(f"Successfully uploaded {len(points)} game vectors to Qdrant.")


if __name__ == "__main__":
    df = load_data()

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = QdrantClient(host="localhost", port=6333)

    setup_qdrant(client, vector_size=model.get_sentence_embedding_dimension())
    embed_and_upload(df, model, client)

    # Sanity check — query with a test game
    test_text = "An open world fantasy RPG with exploration and combat"
    test_vector = model.encode([test_text], normalize_embeddings=True)[0].tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=test_vector, limit=5)
    print("\nSanity check — top 5 results for 'open world fantasy RPG':")
    for r in results.points:
        print(f"  {r.payload['name']} (score: {r.score:.3f})")
# Game Recommender

> AI-powered video game recommendations using hybrid filtering + LLM re-ranking.

## How it works
```
User Input
    │
    ├──► Content-Based Filter  ──────────────────────────┐
    │    (game embeddings via metadata + description)    │
    │                                                    ▼
    ├──► Collaborative Filter  ───────────────►  Fusion/Re-ranking Layer
    │    (Steam playtime matrix factorization)           │
    │                                                    ▼
    └──► LLM Semantic Layer    ──────────────────────────┘
         (natural language query → embedding → RAG)   Final Top-N Results
```
- **Content-based:** Each game's metadata and description is embedded using FastEmbed (`BAAI/bge-small-en-v1.5`) and stored in Qdrant. At query time, the user's input is embedded and the nearest neighbor games are retrieved via cosine similarity, blended with a popularity score based on review count.
- **LLM layer:** The top 20 vector search candidates are passed to Llama 3.3 70B (via Groq) which re-ranks them to a final 5, enforcing that critical requirements are met (mechanics, perspective, genre) and returning a personalized reason for each pick.

## Tech Stack
| Layer | Tool |
|---|---|
| **Backend** | Python + FastAPI |
| **LLM** | Llama 3.3 70B via Groq API |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| **Vector DB** | Qdrant Cloud |
| **Data pipeline** | Python scripts (RAWG API) |
| **Frontend** | React + Tailwind + Vite |
| **Backend hosting** | Render |
| **Frontend hosting** | Vercel |

## Data Pipeline
1. `fetch_games.py` — Fetches ~9k games from the RAWG API
2. `clean_games.py` — Normalizes fields and builds `embedding_text` per game
3. `enrich_games.py` — Fetches full descriptions via RAWG detail endpoint (concurrent)
4. `generate_embeddings.py` — Embeds all games and uploads vectors to Qdrant Cloud


> ⚠️ The backend is hosted on Render's free tier and may take 30 seconds to wake up on first request.

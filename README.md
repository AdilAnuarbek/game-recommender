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
- **Content-based**: Embed each game's metadata + description using sentence-transformers. Store in a vector DB. At query time, embed the user's taste profile and find nearest neighbors.
- **Collaborative**: Train a matrix factorization model (ALS via implicit library) on Steam playtime data. Gives you "users like you also played..." signals.
- **LLM layer**: User's natural language query gets embedded and used to retrieve candidates from the vector DB (RAG), then an LLM re-ranks and justifies the top picks in plain English.

## Tech Stack
- **Backend**: Python + FastAPI
- **LLM**: Claude API (claude-sonnet) via Anthropic SDK
- **Embeddings**: sentence-transformers (all-MiniLM or all-mpnet)
- **Vector DB**: Qdrant (free, self-hostable, great Python SDK)
- **Collab filtering**: implicit (ALS matrix factorization)
- **Data storage**: PostgreSQL (user profiles, feedback)
- **Frontend**: React + Tailwind
- **Deployment**: Railway or Render (backend), Vercel (frontend)
- **Data pipeline**: Python scripts + APScheduler for periodic refresh

## Status
- [x] Repo setup
- [x] Data pipeline (RAWG API)
- [x] Content-based filtering
- [x] Collaborative filtering
- [x] LLM re-ranking layer
- [x] Frontend
- [ ] Deployment
```
The checklist will update as the project progresses and updates
```
> ⚠️ The backend is hosted on Render's free tier and may take 30 seconds to wake up on first request.
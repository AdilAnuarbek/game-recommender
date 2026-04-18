from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from recommender.search import search_games
from recommender.rerank import rerank_with_llm

router = APIRouter()


class RecommendRequest(BaseModel):
    query: str                        # natural language input
    liked_games: list[str] = []       # game names the user likes
    top_k: int = 20                   # candidates from vector search
    final_k: int = 5                  # results after LLM re-ranking


class GameResult(BaseModel):
    name: str
    genres: str
    tags: str
    rating: float
    playtime: float
    released: str
    background_image: str
    slug: str
    reason: str                       # LLM-generated personalized reason


class RecommendResponse(BaseModel):
    recommendations: list[GameResult]


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if not request.query and not request.liked_games:
        raise HTTPException(status_code=400, detail="Provide a query or at least one liked game.")

    # Step 1 — vector search for candidates
    candidates = search_games(
        query=request.query,
        liked_games=request.liked_games,
        top_k=request.top_k,
    )

    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates found.")

    # Step 2 — LLM re-ranking
    recommendations = rerank_with_llm(
        query=request.query,
        liked_games=request.liked_games,
        candidates=candidates,
        final_k=request.final_k,
    )

    return RecommendResponse(recommendations=recommendations)
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, g in enumerate(candidates, 1):
        lines.append(
            f"{i}. {g['name']} | Genres: {g['genres']} | "
            f"Tags: {g['tags']} | Rating: {g['rating']} | "
            f"Playtime: {g['playtime']}h | Released: {g['released']}"
        )
    return "\n".join(lines)


def rerank_with_llm(
    query: str,
    liked_games: list[str],
    candidates: list[dict],
    final_k: int = 5,
) -> list[dict]:

    prompt = f"""You are a strict video game recommendation expert.

User's request: "{query}"
Games the user enjoys: {", ".join(liked_games) if liked_games else "Not specified"}

Here are {len(candidates)} candidate games:
{format_candidates(candidates)}

Your job is to pick the best {final_k} games that CLOSELY match the user's request.

Rules:
- If the user mentions specific mechanics (e.g. guns, building, zombies), the game MUST have them.
- If the user mentions a perspective (e.g. 3rd person), strongly prefer games with that perspective.
- Do NOT recommend a game that is missing critical elements, even if it matches partially.
- Do NOT invent or assume features a game doesn't have to make it fit.
- If fewer than {final_k} games genuinely match, return only the ones that do. Do not pad with weak matches.

Return ONLY a JSON array, no extra text:
[
  {{
    "name": "Game Name",
    "reason": "One sentence explaining specifically how this game matches the user's request."
  }}
]"""

    message = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    ranked = json.loads(raw)

    # Merge LLM picks back with full candidate data
    candidate_map = {g["name"].lower(): g for g in candidates}
    results = []
    for pick in ranked[:final_k]:
        name_key = pick["name"].lower()
        base = candidate_map.get(name_key, {})
        results.append({
            "name":             pick["name"],
            "genres":           base.get("genres", ""),
            "tags":             base.get("tags", ""),
            "rating":           base.get("rating", 0.0),
            "playtime":         base.get("playtime", 0.0),
            "released":         base.get("released", ""),
            "background_image": base.get("background_image", ""),
            "slug":             base.get("slug", ""),   # add this
            "reason":           pick["reason"],
        })

    return results
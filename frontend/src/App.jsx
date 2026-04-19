import { useState } from "react"
import axios from "axios"

const API_URL = import.meta.env.VITE_API_URL ?? "https://game-recommender-dboq.onrender.com/api/recommend"

export default function App() {
  const [query, setQuery] = useState("")
  const [likedInput, setLikedInput] = useState("")
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async () => {
    if (!query && !likedInput) return
    setLoading(true)
    setError("")
    setResults([])

    const liked_games = likedInput
      .split(",")
      .map(g => g.trim())
      .filter(Boolean)

    try {
      const res = await axios.post(API_URL, {
        query,
        liked_games,
        top_k: 20,
        final_k: 5,
      })
      setResults(res.data.recommendations)
    } catch (e) {
      setError("Something went wrong. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white px-6 py-12 max-w-3xl mx-auto">
      <h1 className="text-4xl font-bold mb-2">🎮 Game Recommender</h1>
      <p className="text-gray-400 mb-8">Describe what you're in the mood for and get AI-powered picks.</p>

      <div className="flex flex-col gap-4 mb-8">
        <textarea
          className="bg-gray-800 rounded-xl p-4 text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
          rows={3}
          placeholder='e.g. "An open world RPG with great story and dark atmosphere"'
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <input
          className="bg-gray-800 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          placeholder='Games you like (comma-separated): e.g. "Elden Ring, The Witcher 3"'
          value={likedInput}
          onChange={e => setLikedInput(e.target.value)}
        />
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-xl py-3 font-semibold transition-colors"
        >
          {loading ? "Finding games..." : "Recommend Games"}
        </button>
      </div>

      {error && <p className="text-red-400 mb-6">{error}</p>}

      <div className="flex flex-col gap-6">
        {results.map((game, i) => (
          <div key={i} className="bg-gray-800 rounded-2xl overflow-hidden flex gap-4">
            {game.background_image && (
              <img
                src={game.background_image}
                alt={game.name}
                className="w-40 h-28 object-cover shrink-0"
              />
            )}
            <div className="py-4 pr-4">
              <div className="flex items-center gap-3 mb-1">
            <a href={`https://rawg.io/games/${game.slug}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-lg font-bold hover:text-indigo-400 transition-colors">
              {game.name}
            </a>
                <span className="text-yellow-400 text-sm">★ {game.rating?.toFixed(1)}</span>
                {game.released && (
                  <span className="text-gray-500 text-sm">{game.released.slice(0, 4)}</span>
                )}
              </div>
              <p className="text-gray-400 text-sm mb-2">{game.genres}</p>
              <p className="text-indigo-300 text-sm italic">"{game.reason}"</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
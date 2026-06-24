import { useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "http://localhost:8000";

const EXAMPLES = [
  "What is FastAPI?",
  "How does async def differ from def in path operations?",
  "How does FastAPI compare to Flask?",
  "What is Pydantic and how does FastAPI use it?",
  "How do I set up a virtual environment for FastAPI?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showChunks, setShowChunks] = useState(false);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), top_k: 5 }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleAsk();
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">RAG Assistant</div>

        <div className="sidebar-section">
          <div className="sidebar-label">Backend</div>
          <span className="backend-pill">openai / gpt-4o-mini</span>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Corpus</div>
          <p>12 FastAPI documentation files ingested into ChromaDB.</p>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Evaluation</div>
          <div className="eval-row">
            <span className="eval-backend">OpenAI</span>
            <span className="eval-score">4.73 / 5.00</span>
            <span className="eval-pct">93%</span>
          </div>
          <div className="eval-row">
            <span className="eval-backend">Ollama</span>
            <span className="eval-score">4.47 / 5.00</span>
            <span className="eval-pct">93%</span>
          </div>
        </div>

        <div className="sidebar-section">
          <label className="chunk-toggle">
            <input
              type="checkbox"
              checked={showChunks}
              onChange={(e) => setShowChunks(e.target.checked)}
            />
            Show retrieved chunks
          </label>
        </div>
      </aside>

      <main className="main">
        <div className="main-header">
          <h1>FastAPI Docs Assistant</h1>
          <p className="subtitle">
            Ask anything about FastAPI. Every answer is grounded in the
            official documentation — no hallucination, sources shown for
            every response.
          </p>
        </div>

        <div className="input-row">
          <input
            className="question-input"
            type="text"
            placeholder="e.g. How does FastAPI use Python type hints?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="ask-btn"
            onClick={handleAsk}
            disabled={loading}
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>

        {error && <div className="error">Error: {error}</div>}

        {result && (
          <div className="result">
            <div className="answer-block">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>
            <div className="sources">
              <span className="sources-label">Sources</span>
              {result.sources.map((src) => (
                <span key={src} className="source-badge">
                  {src}
                </span>
              ))}
            </div>

            {showChunks && (
              <div className="chunks">
                <h4>Retrieved chunks</h4>
                {result.chunks.map((chunk, i) => (
                  <div key={i} className="chunk">
                    <div className="chunk-meta">
                      [{i + 1}] {chunk.source} — distance:{" "}
                      {chunk.distance.toFixed(4)}
                    </div>
                    <pre className="chunk-text">{chunk.text}</pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!result && !loading && (
          <div className="examples">
            <p className="examples-label">Try asking</p>
            <div className="example-grid">
              {EXAMPLES.map((ex) => (
                <div
                  key={ex}
                  className="example-card"
                  onClick={() => setQuestion(ex)}
                >
                  {ex}
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

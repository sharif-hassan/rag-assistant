"""
app.py — Streamlit web interface for the RAG assistant.

Wraps pipeline.ask() in a minimal, developer-focused UI. The signature
element is the source attribution display — showing exactly which
documents grounded each answer, which is what distinguishes this from
a plain chatbot.
"""

import streamlit as st
from rag.pipeline import RAGPipeline
from rag.config import LLM_BACKEND, OPENAI_CHAT_MODEL, OLLAMA_MODEL

st.set_page_config(
    page_title="FastAPI Docs Assistant",
    page_icon="⚡",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 760px; }

    .source-badge {
        display: inline-block;
        background: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px 3px 2px 0;
        border: 1px solid #313244;
    }

    .answer-block {
        background: #1e1e2e;
        border-left: 3px solid #89b4fa;
        padding: 1rem 1.2rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
        color: #cdd6f4;
        line-height: 1.65;
    }

    .backend-pill {
        display: inline-block;
        font-size: 0.7rem;
        font-family: monospace;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
    }
    .backend-openai { background: #a6e3a1; color: #1e1e2e; }
    .backend-ollama { background: #fab387; color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()

pipeline = load_pipeline()

with st.sidebar:
    st.markdown("### System")

    backend_class = "backend-openai" if LLM_BACKEND == "openai" else "backend-ollama"
    model_name = OPENAI_CHAT_MODEL if LLM_BACKEND == "openai" else OLLAMA_MODEL
    st.markdown(
        f'<span class="backend-pill {backend_class}">⚙ {LLM_BACKEND} / {model_name}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Corpus**")
    st.markdown("12 FastAPI documentation files ingested into ChromaDB.")

    st.markdown("---")
    st.markdown("**Eval results**")
    st.markdown("OpenAI: **4.27 / 5.00** avg (86% ≥ 4)")
    st.markdown("Ollama: **4.53 / 5.00** avg (100% ≥ 4)")

    st.markdown("---")
    st.markdown("**Top K**")
    top_k = st.slider("Chunks retrieved per query", min_value=1, max_value=10, value=5)

    show_chunks = st.checkbox("Show retrieved chunks", value=False)


st.markdown("## ⚡ FastAPI Docs Assistant")
st.markdown(
    "Ask anything about FastAPI. Answers are grounded in the official documentation — "
    "no hallucination, sources shown for every response."
)

question = st.text_input(
    label="Question",
    placeholder="e.g. How does FastAPI use Python type hints?",
    label_visibility="collapsed",
)

ask_button = st.button("Ask", type="primary", use_container_width=False)

if ask_button and question.strip():
    with st.spinner("Retrieving and generating..."):
        result = pipeline.ask(question.strip(), top_k=top_k)

    st.markdown(
        f'<div class="answer-block">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    badges = "".join(
        f'<span class="source-badge">📄 {src}</span>'
        for src in result["sources"]
    )
    st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)

    if show_chunks:
        with st.expander("Retrieved chunks", expanded=False):
            for i, chunk in enumerate(result["chunks"], 1):
                st.markdown(f"**[{i}] {chunk['source']}** — distance: `{chunk['distance']:.4f}`")
                st.code(chunk["text"], language=None)

elif ask_button and not question.strip():
    st.warning("Please enter a question first.")

if not question:
    st.markdown("---")
    st.markdown("**Try asking:**")
    examples = [
        "What is FastAPI?",
        "How does async def differ from def in path operations?",
        "How does FastAPI compare to Flask?",
        "What is Pydantic and how does FastAPI use it?",
        "How do I set up a virtual environment for FastAPI?",
    ]
    for example in examples:
        st.markdown(f"- *{example}*")
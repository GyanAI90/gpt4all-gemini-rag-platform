"""
GPT4All + Gemini RAG Intelligence Platform
==========================================

Streamlit UI for the dual-model RAG platform.

Run:
    python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# BACKEND
# ============================================================

from src.dual_rag_pipeline import DualRAGPipeline
from src.evaluator import (
    evaluate_retrieval,
    evaluate_groundedness,
    evaluate_source_coverage,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="RAG Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background-color: #0e1117;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

section[data-testid="stSidebar"] {
    background-color: #11151d;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

section[data-testid="stSidebar"] hr {
    border-color: #2a3140;
}

.hero-box {
    background: linear-gradient(135deg, #171c27 0%, #10141c 100%);
    border: 1px solid #2a3140;
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 24px;
}

.hero-title {
    color: #f3f4f6;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 15px;
    line-height: 1.6;
}

.section-title {
    color: #f3f4f6;
    font-size: 20px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 12px;
}

.metric-card {
    background-color: #151922;
    border: 1px solid #2a3140;
    border-radius: 12px;
    padding: 16px;
    min-height: 88px;
}

.metric-label {
    color: #9ca3af;
    font-size: 12px;
    margin-bottom: 7px;
}

.metric-value {
    color: #f3f4f6;
    font-size: 23px;
    font-weight: 700;
}

.source-card {
    background-color: #151922;
    border: 1px solid #2a3140;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.source-title {
    color: #e5e7eb;
    font-weight: 600;
    font-size: 14px;
}

.source-meta {
    color: #9ca3af;
    font-size: 12px;
    margin-top: 5px;
}

.empty-state {
    background-color: #151922;
    border: 1px solid #2a3140;
    border-radius: 16px;
    padding: 55px 25px;
    text-align: center;
    margin-top: 20px;
}

.empty-icon {
    font-size: 46px;
    margin-bottom: 12px;
}

.empty-title {
    color: #f3f4f6;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 8px;
}

.empty-text {
    color: #9ca3af;
    font-size: 15px;
}

.status-ok {
    color: #34d399;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return DualRAGPipeline(
        top_k=5,
        enable_gemini=True,
    )


# ============================================================
# HELPERS
# ============================================================

def format_score(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def format_seconds(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "N/A"


def get_score(document):
    for key in ("similarity", "similarity_score"):
        if key in document:
            return document[key]

    if "distance" in document:
        try:
            distance = float(document["distance"])
            return 1.0 / (1.0 + distance)
        except (TypeError, ValueError):
            pass

    return None


def metric_card(label, value):
    st.markdown(
        f"""
<div class="metric-card">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sources(sources):
    if not sources:
        st.info("No source information available.")
        return

    for i, source in enumerate(sources, 1):
        document = source.get("source", "Unknown")
        page = source.get("page", "Unknown")
        chunk = source.get("chunk_id", "Unknown")

        st.markdown(
            f"""
<div class="source-card">
<div class="source-title">📄 Source {i} — {document}</div>
<div class="source-meta">Page: {page} &nbsp;|&nbsp; Chunk: {chunk}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_retrieved_documents(documents):
    if not documents:
        st.info("No documents were retrieved.")
        return

    for i, document in enumerate(documents, 1):
        metadata = document.get("metadata", {})

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        chunk_id = metadata.get("chunk_id", "Unknown")
        score = get_score(document)

        with st.expander(
            f"Source {i} — {source} — Page {page}"
        ):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption(f"Page: {page}")

            with c2:
                st.caption(f"Chunk: {chunk_id}")

            with c3:
                st.caption(
                    f"Similarity: {format_score(score)}"
                )

            st.write(
                document.get("content", "")
            )


def render_timing(timing):
    items = [
        ("Retrieval", "retrieval_seconds"),
        ("Context", "context_seconds"),
        ("Prompt", "prompt_seconds"),
        ("GPT4All", "gpt4all_seconds"),
        ("Gemini", "gemini_seconds"),
        ("Total", "total_seconds"),
    ]

    columns = st.columns(6)

    for column, (label, key) in zip(columns, items):
        with column:
            st.metric(
                label,
                format_seconds(timing.get(key)),
            )


def evaluate_result(question, result):
    documents = result.get("retrieved_documents", [])
    sources = result.get("sources", [])
    context = result.get("context", "")

    gpt4all_answer = result.get("gpt4all_answer", "")
    gemini_answer = result.get("gemini_answer")

    try:
        retrieval = evaluate_retrieval(
            question,
            documents,
        )
    except Exception:
        retrieval = {
            "average_keyword_relevance": 0.0
        }

    try:
        gpt4all_grounding = evaluate_groundedness(
            gpt4all_answer,
            context,
        )
    except Exception:
        gpt4all_grounding = {
            "groundedness_score": 0.0
        }

    try:
        gpt4all_coverage = evaluate_source_coverage(
            gpt4all_answer,
            sources,
        )
    except Exception:
        gpt4all_coverage = {
            "source_coverage": 0.0,
            "cited_sources": 0,
            "total_sources": len(sources),
        }

    gemini_grounding = None
    gemini_coverage = None

    if gemini_answer:
        try:
            gemini_grounding = evaluate_groundedness(
                gemini_answer,
                context,
            )
        except Exception:
            gemini_grounding = {
                "groundedness_score": 0.0
            }

        try:
            gemini_coverage = evaluate_source_coverage(
                gemini_answer,
                sources,
            )
        except Exception:
            gemini_coverage = {
                "source_coverage": 0.0,
                "cited_sources": 0,
                "total_sources": len(sources),
            }

    return (
        retrieval,
        gpt4all_grounding,
        gpt4all_coverage,
        gemini_grounding,
        gemini_coverage,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 RAG Intelligence")

    st.caption(
        "Local + Cloud Retrieval Augmented Generation"
    )

    st.divider()

    st.markdown("### Model")

    model_mode = st.radio(
        "Select inference mode",
        ["GPT4All", "Gemini", "Compare Both"],
        index=0,
    )

    st.divider()

    st.markdown("### Knowledge Base")

    st.markdown(
        '<div class="status-ok">● Vector database available</div>',
        unsafe_allow_html=True,
    )

    st.write("Documents: 3")
    st.write("Pages: 254")
    st.write("Chunks: 1,545")
    st.write("Embedding: MiniLM-L6-v2")
    st.write("Dimensions: 384")
    st.write("Retriever: Top-K 5")

    st.divider()

    st.markdown("### LLM Configuration")

    st.write("Local: Qwen2-1.5B")
    st.write("Cloud: Gemini 3.6 Flash")

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-box">
<div class="hero-title">🧠 Local RAG Intelligence Platform</div>
<div class="hero-subtitle">
GPT4All + Gemini + ChromaDB
&nbsp;•&nbsp;
Document Intelligence
&nbsp;•&nbsp;
Source Grounding
&nbsp;•&nbsp;
Model Comparison
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE LOAD
# ============================================================

try:
    with st.spinner("Loading RAG pipeline..."):
        pipeline = load_pipeline()
except Exception as error:
    st.error("Failed to initialize the RAG pipeline.")
    st.exception(error)
    st.stop()


# ============================================================
# PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
<div class="empty-state">
<div class="empty-icon">🧠</div>
<div class="empty-title">Ask your documents anything</div>
<div class="empty-text">
Query your knowledge base using GPT4All, Gemini, or compare both models.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving documents and generating answer..."
        ):

            try:
                result = pipeline.ask(question)
                st.session_state.last_result = result

            except Exception as error:
                st.error("RAG pipeline failed.")
                st.exception(error)
                st.stop()

        documents = result.get(
            "retrieved_documents",
            [],
        )

        sources = result.get(
            "sources",
            [],
        )

        context = result.get(
            "context",
            "",
        )

        timing = result.get(
            "timing",
            {},
        )

        gpt4all_answer = result.get(
            "gpt4all_answer",
            "",
        )

        gemini_answer = result.get(
            "gemini_answer"
        )

        (
            retrieval_eval,
            gpt4all_grounding,
            gpt4all_coverage,
            gemini_grounding,
            gemini_coverage,
        ) = evaluate_result(
            question,
            result,
        )

        # ----------------------------------------------------
        # ANSWERS
        # ----------------------------------------------------

        if model_mode in ("GPT4All", "Compare Both"):

            st.markdown(
                "### 🟢 GPT4All / Qwen2-1.5B"
            )

            st.markdown(gpt4all_answer)

        if model_mode in ("Gemini", "Compare Both"):

            st.markdown(
                "### 🔵 Gemini"
            )

            if gemini_answer:
                st.markdown(gemini_answer)
            else:
                st.warning(
                    "Gemini is currently unavailable."
                )

                if result.get("gemini_error"):
                    st.caption(
                        result["gemini_error"]
                    )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Evaluation</div>',
            unsafe_allow_html=True,
        )

        if model_mode == "GPT4All":

            columns = st.columns(5)

            with columns[0]:
                metric_card(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:
                metric_card(
                    "Relevance",
                    format_score(
                        retrieval_eval.get(
                            "average_keyword_relevance"
                        )
                    ),
                )

            with columns[2]:
                metric_card(
                    "Groundedness",
                    format_score(
                        gpt4all_grounding.get(
                            "groundedness_score"
                        )
                    ),
                )

            with columns[3]:
                metric_card(
                    "Generation",
                    format_seconds(
                        timing.get(
                            "gpt4all_seconds"
                        )
                    ),
                )

            with columns[4]:
                metric_card(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        elif model_mode == "Gemini":

            columns = st.columns(5)

            with columns[0]:
                metric_card(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:
                metric_card(
                    "Relevance",
                    format_score(
                        retrieval_eval.get(
                            "average_keyword_relevance"
                        )
                    ),
                )

            with columns[2]:
                metric_card(
                    "Groundedness",
                    format_score(
                        gemini_grounding.get(
                            "groundedness_score"
                        )
                    )
                    if gemini_grounding
                    else "N/A",
                )

            with columns[3]:
                metric_card(
                    "Generation",
                    format_seconds(
                        timing.get(
                            "gemini_seconds"
                        )
                    ),
                )

            with columns[4]:
                metric_card(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        else:

            columns = st.columns(5)

            with columns[0]:
                metric_card(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:
                metric_card(
                    "Relevance",
                    format_score(
                        retrieval_eval.get(
                            "average_keyword_relevance"
                        )
                    ),
                )

            with columns[2]:
                metric_card(
                    "Qwen Grounding",
                    format_score(
                        gpt4all_grounding.get(
                            "groundedness_score"
                        )
                    ),
                )

            with columns[3]:
                metric_card(
                    "Gemini Grounding",
                    format_score(
                        gemini_grounding.get(
                            "groundedness_score"
                        )
                    )
                    if gemini_grounding
                    else "N/A",
                )

            with columns[4]:
                metric_card(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        with st.expander("⏱️ Pipeline Timing"):
            render_timing(timing)

        st.markdown(
            '<div class="section-title">📚 Sources</div>',
            unsafe_allow_html=True,
        )

        render_sources(sources)

        with st.expander("🔎 Retrieved Chunks"):
            render_retrieved_documents(documents)

        with st.expander("📄 Shared RAG Context"):
            st.text(context)

        with st.expander("📌 Source Coverage"):

            left, right = st.columns(2)

            with left:
                st.markdown("#### GPT4All")

                st.metric(
                    "Coverage",
                    format_score(
                        gpt4all_coverage.get(
                            "source_coverage"
                        )
                    ),
                )

                st.caption(
                    f"Cited sources: "
                    f"{gpt4all_coverage.get('cited_sources', 0)}"
                    f"/"
                    f"{gpt4all_coverage.get('total_sources', len(sources))}"
                )

            with right:
                st.markdown("#### Gemini")

                if gemini_coverage:

                    st.metric(
                        "Coverage",
                        format_score(
                            gemini_coverage.get(
                                "source_coverage"
                            )
                        ),
                    )

                    st.caption(
                        f"Cited sources: "
                        f"{gemini_coverage.get('cited_sources', 0)}"
                        f"/"
                        f"{gemini_coverage.get('total_sources', len(sources))}"
                    )

                else:
                    st.info(
                        "Gemini answer unavailable."
                    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if model_mode == "GPT4All":

        assistant_text = gpt4all_answer

    elif model_mode == "Gemini":

        assistant_text = (
            gemini_answer
            or "Gemini unavailable."
        )

    else:

        assistant_text = (
            "### GPT4All / Qwen2-1.5B\n\n"
            + gpt4all_answer
        )

        if gemini_answer:
            assistant_text += (
                "\n\n### Gemini\n\n"
                + gemini_answer
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )

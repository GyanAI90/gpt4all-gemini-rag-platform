"""
GPT4All + Gemini RAG Platform
=============================

Professional Streamlit UI for the dual-model RAG system.

Features:
    - GPT4All local inference
    - Gemini cloud inference
    - Dual-model comparison
    - ChromaDB retrieval
    - Source exploration
    - RAG evaluation metrics
    - Chat history
    - Pipeline timing
    - Error handling
"""

import sys
import time
from pathlib import Path

import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ============================================================
# IMPORT BACKEND
# ============================================================

from dual_rag_pipeline import DualRAGPipeline
from evaluator import (
    evaluate_retrieval,
    evaluate_groundedness,
    evaluate_source_coverage,
    answer_statistics,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* ------------------------------------------------------
       GLOBAL
       ------------------------------------------------------ */

    .stApp {
        background-color: #0e1117;
    }

    .main {
        background-color: #0e1117;
    }

    /* ------------------------------------------------------
       HEADER
       ------------------------------------------------------ */

    .hero {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #171b26,
            #11151d
        );
        border: 1px solid #262c38;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* ------------------------------------------------------
       METRIC CARDS
       ------------------------------------------------------ */

    .metric-card {
        background: #151922;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 1rem;
        min-height: 110px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
    }

    /* ------------------------------------------------------
       ANSWER CARD
       ------------------------------------------------------ */

    .answer-card {
        background: #151922;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .answer-header {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    /* ------------------------------------------------------
       SOURCE CARD
       ------------------------------------------------------ */

    .source-card {
        background: #131720;
        border: 1px solid #292f3a;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.6rem;
    }

    .source-title {
        font-weight: 600;
    }

    .source-meta {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    /* ------------------------------------------------------
       STATUS
       ------------------------------------------------------ */

    .status-ok {
        color: #34d399;
        font-weight: 600;
    }

    .status-warning {
        color: #fbbf24;
        font-weight: 600;
    }

    .status-error {
        color: #f87171;
        font-weight: 600;
    }

    /* ------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        background-color: #11151d;
    }

    /* ------------------------------------------------------
       DIVIDER
       ------------------------------------------------------ */

    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
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


if "pipeline" not in st.session_state:

    st.session_state.pipeline = None


# ============================================================
# CACHED PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_pipeline():

    return DualRAGPipeline(
        top_k=5,
        enable_gemini=True,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_score(
    value,
):
    """Format a numeric score."""

    if value is None:
        return "N/A"

    return f"{value:.3f}"


def format_seconds(
    value,
):
    """Format latency."""

    if value is None:
        return "N/A"

    return f"{value:.2f}s"


def get_retrieval_score(
    result,
):
    """
    Get similarity score from a retrieved result.

    Supports different metadata structures used by
    ChromaDB/retriever implementations.
    """

    if "similarity" in result:
        return result["similarity"]

    if "similarity_score" in result:
        return result["similarity_score"]

    if "distance" in result:

        distance = result["distance"]

        try:
            return 1 / (1 + distance)
        except Exception:
            return None

    return None


def render_metric(
    label,
    value,
):
    """Render a metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(
    sources,
):
    """Render retrieved source cards."""

    if not sources:

        st.info(
            "No source information available."
        )

        return

    for index, source in enumerate(
        sources,
        start=1,
    ):

        document = source.get(
            "source",
            "Unknown",
        )

        page = source.get(
            "page",
            "Unknown",
        )

        chunk = source.get(
            "chunk_id",
            "Unknown",
        )

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    📄 Source {index} — {document}
                </div>

                <div class="source-meta">
                    Page: {page}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Chunk: {chunk}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


def render_retrieved_documents(
    documents,
):
    """Display retrieved chunks with similarity."""

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown",
        )

        score = get_retrieval_score(
            document
        )

        with st.expander(
            f"Source {index} — {source} — Page {page}"
        ):

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.caption(
                    f"Page: {page}"
                )

            with col2:

                st.caption(
                    f"Chunk: {chunk_id}"
                )

            with col3:

                st.caption(
                    f"Similarity: "
                    f"{format_score(score)}"
                )

            st.write(
                document.get(
                    "content",
                    "",
                )
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 RAG Intelligence"
    )

    st.caption(
        "Local + Cloud Retrieval Augmented Generation"
    )

    st.divider()

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    st.markdown(
        "### Model"
    )

    model_mode = st.radio(
        "Select inference mode",
        [
            "GPT4All",
            "Gemini",
            "Compare Both",
        ],
        index=0,
    )

    st.divider()

    # --------------------------------------------------------
    # Knowledge base
    # --------------------------------------------------------

    st.markdown(
        "### Knowledge Base"
    )

    st.markdown(
        """
        <div class="status-ok">
        ● Vector database available
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Documents: 3"
    )

    st.write(
        "Pages: 254"
    )

    st.write(
        "Chunks: 1,545"
    )

    st.write(
        "Embedding: MiniLM-L6-v2"
    )

    st.write(
        "Dimensions: 384"
    )

    st.write(
        "Retriever: Top-K 5"
    )

    st.divider()

    # --------------------------------------------------------
    # LLM information
    # --------------------------------------------------------

    st.markdown(
        "### LLM Configuration"
    )

    st.write(
        "Local: Qwen2-1.5B"
    )

    st.write(
        "Cloud: Gemini 3.6 Flash"
    )

    st.divider()

    # --------------------------------------------------------
    # Chat controls
    # --------------------------------------------------------

    st.markdown(
        "### Chat"
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.last_result = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 Local RAG Intelligence Platform
        </div>

        <div class="hero-subtitle">
            GPT4All + Gemini + ChromaDB
            • Document Intelligence
            • Source Grounding
            • Model Comparison
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE LOADING
# ============================================================

try:

    if st.session_state.pipeline is None:

        with st.spinner(
            "Loading RAG pipeline..."
        ):

            st.session_state.pipeline = (
                load_pipeline()
            )

    pipeline = st.session_state.pipeline

except Exception as error:

    st.error(
        "Failed to initialize the RAG pipeline."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Retrieving documents and generating answer..."
        ):

            try:

                start_time = (
                    time.perf_counter()
                )

                result = pipeline.ask(
                    question
                )

                ui_total_time = (
                    time.perf_counter()
                    - start_time
                )

                st.session_state.last_result = (
                    result
                )

            except Exception as error:

                st.error(
                    "RAG pipeline failed."
                )

                st.exception(
                    error
                )

                st.stop()

        # ----------------------------------------------------
        # Answers
        # ----------------------------------------------------

        if model_mode in (
            "GPT4All",
            "Compare Both",
        ):

            st.markdown(
                "### 🟢 GPT4All / Qwen2-1.5B"
            )

            st.markdown(
                f"""
                <div class="answer-card">
                    {result['gpt4all_answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if model_mode in (
            "Gemini",
            "Compare Both",
        ):

            st.markdown(
                "### 🔵 Gemini"
            )

            if result.get(
                "gemini_answer"
            ):

                st.markdown(
                    f"""
                    <div class="answer-card">
                        {result['gemini_answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.warning(
                    "Gemini is currently unavailable."
                )

                if result.get(
                    "gemini_error"
                ):

                    st.caption(
                        result[
                            "gemini_error"
                        ]
                    )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Evaluation"
        )

        documents = result[
            "retrieved_documents"
        ]

        sources = result[
            "sources"
        ]

        context = result[
            "context"
        ]

        retrieval_eval = (
            evaluate_retrieval(
                question,
                documents,
            )
        )

        gpt4all_eval = (
            evaluate_groundedness(
                result[
                    "gpt4all_answer"
                ],
                context,
            )
        )

        gpt4all_source_eval = (
            evaluate_source_coverage(
                result[
                    "gpt4all_answer"
                ],
                sources,
            )
        )

        gemini_eval = None

        gemini_source_eval = None

        if result.get(
            "gemini_answer"
        ):

            gemini_eval = (
                evaluate_groundedness(
                    result[
                        "gemini_answer"
                    ],
                    context,
                )
            )

            gemini_source_eval = (
                evaluate_source_coverage(
                    result[
                        "gemini_answer"
                    ],
                    sources,
                )
            )

        timing = result.get(
            "timing",
            {},
        )

        # ----------------------------------------------------
        # Metric cards
        # ----------------------------------------------------

        if model_mode == "GPT4All":

            columns = st.columns(
                5
            )

            with columns[0]:

                render_metric(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:

                render_metric(
                    "Relevance",
                    format_score(
                        retrieval_eval[
                            "average_keyword_relevance"
                        ]
                    ),
                )

            with columns[2]:

                render_metric(
                    "Groundedness",
                    format_score(
                        gpt4all_eval[
                            "groundedness_score"
                        ]
                    ),
                )

            with columns[3]:

                render_metric(
                    "Generation",
                    format_seconds(
                        timing.get(
                            "gpt4all_seconds"
                        )
                    ),
                )

            with columns[4]:

                render_metric(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        elif model_mode == "Gemini":

            columns = st.columns(
                5
            )

            with columns[0]:

                render_metric(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:

                render_metric(
                    "Relevance",
                    format_score(
                        retrieval_eval[
                            "average_keyword_relevance"
                        ]
                    ),
                )

            with columns[2]:

                render_metric(
                    "Groundedness",
                    format_score(
                        gemini_eval[
                            "groundedness_score"
                        ]
                    )
                    if gemini_eval
                    else "N/A",
                )

            with columns[3]:

                render_metric(
                    "Generation",
                    format_seconds(
                        timing.get(
                            "gemini_seconds"
                        )
                    ),
                )

            with columns[4]:

                render_metric(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        else:

            columns = st.columns(
                5
            )

            with columns[0]:

                render_metric(
                    "Retrieved",
                    len(documents),
                )

            with columns[1]:

                render_metric(
                    "Relevance",
                    format_score(
                        retrieval_eval[
                            "average_keyword_relevance"
                        ]
                    ),
                )

            with columns[2]:

                render_metric(
                    "Qwen Grounding",
                    format_score(
                        gpt4all_eval[
                            "groundedness_score"
                        ]
                    ),
                )

            with columns[3]:

                render_metric(
                    "Gemini Grounding",
                    format_score(
                        gemini_eval[
                            "groundedness_score"
                        ]
                    )
                    if gemini_eval
                    else "N/A",
                )

            with columns[4]:

                render_metric(
                    "Total",
                    format_seconds(
                        timing.get(
                            "total_seconds"
                        )
                    ),
                )

        # ----------------------------------------------------
        # Pipeline timing
        # ----------------------------------------------------

        with st.expander(
            "⏱️ Pipeline Timing"
        ):

            timing_columns = st.columns(
                6
            )

            timing_items = [
                (
                    "Retrieval",
                    timing.get(
                        "retrieval_seconds"
                    ),
                ),
                (
                    "Context",
                    timing.get(
                        "context_seconds"
                    ),
                ),
                (
                    "Prompt",
                    timing.get(
                        "prompt_seconds"
                    ),
                ),
                (
                    "GPT4All",
                    timing.get(
                        "gpt4all_seconds"
                    ),
                ),
                (
                    "Gemini",
                    timing.get(
                        "gemini_seconds"
                    ),
                ),
                (
                    "Total",
                    timing.get(
                        "total_seconds"
                    ),
                ),
            ]

            for column, (
                label,
                value,
            ) in zip(
                timing_columns,
                timing_items,
            ):

                with column:

                    st.metric(
                        label,
                        format_seconds(
                            value
                        ),
                    )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        st.markdown(
            "### 📚 Sources"
        )

        render_sources(
            sources
        )

        # ----------------------------------------------------
        # Retrieved chunks
        # ----------------------------------------------------

        with st.expander(
            "🔎 Retrieved Chunks"
        ):

            render_retrieved_documents(
                documents
            )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        with st.expander(
            "📄 Shared RAG Context"
        ):

            st.text(
                context
            )

        # ----------------------------------------------------
        # Source evaluation
        # ----------------------------------------------------

        with st.expander(
            "📌 Source Coverage"
        ):

            source_columns = st.columns(
                2
            )

            with source_columns[0]:

                st.markdown(
                    "#### GPT4All"
                )

                st.write(
                    f"Coverage: "
                    f"{gpt4all_source_eval['source_coverage']:.3f}"
                )

                st.write(
                    f"Cited sources: "
                    f"{gpt4all_source_eval['cited_sources']}"
                    f"/"
                    f"{gpt4all_source_eval['total_sources']}"
                )

            with source_columns[1]:

                st.markdown(
                    "#### Gemini"
                )

                if gemini_source_eval:

                    st.write(
                        f"Coverage: "
                        f"{gemini_source_eval['source_coverage']:.3f}"
                    )

                    st.write(
                        f"Cited sources: "
                        f"{gemini_source_eval['cited_sources']}"
                        f"/"
                        f"{gemini_source_eval['total_sources']}"
                    )

                else:

                    st.info(
                        "Gemini answer unavailable."
                    )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    if model_mode == "GPT4All":

        assistant_text = (
            result[
                "gpt4all_answer"
            ]
        )

    elif model_mode == "Gemini":

        assistant_text = (
            result.get(
                "gemini_answer"
            )
            or "Gemini unavailable."
        )

    else:

        assistant_text = (
            "GPT4All:\n\n"
            + result[
                "gpt4all_answer"
            ]
        )

        if result.get(
            "gemini_answer"
        ):

            assistant_text += (
                "\n\nGemini:\n\n"
                + result[
                    "gemini_answer"
                ]
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:3rem 1rem;
            color:#9ca3af;
        ">

            <div style="
                font-size:3rem;
                margin-bottom:1rem;
            ">
                🧠
            </div>

            <h2>
                Ask your documents anything
            </h2>

            <p>
                Query your local knowledge base using
                GPT4All, Gemini, or compare both models.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )
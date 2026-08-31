"""
RAG Evaluation Module
=====================

Evaluates the dual-model RAG pipeline.

Models:
    - GPT4All / Qwen2-1.5B
    - Gemini

Evaluation:
    - Retrieval relevance
    - Groundedness
    - Source coverage
    - Answer statistics
    - Latency
    - Model comparison
"""

import re
import time
from typing import Dict, Any, List, Set

from dual_rag_pipeline import DualRAGPipeline


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def get_words(text: str) -> Set[str]:
    """Return normalized words longer than two characters."""

    normalized = normalize_text(text)

    return {
        word
        for word in normalized.split()
        if len(word) > 2
    }


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def evaluate_retrieval(
    question: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Estimate retrieval relevance using keyword overlap.

    This is a deterministic heuristic, not a semantic
    relevance judge.
    """

    question_words = get_words(
        question
    )

    document_scores = []

    for result in results:

        content = result.get(
            "content",
            "",
        )

        document_words = get_words(
            content
        )

        if not question_words:

            overlap_score = 0.0

        else:

            overlap_score = (
                len(
                    question_words
                    & document_words
                )
                / len(question_words)
            )

        document_scores.append(
            overlap_score
        )

    if document_scores:

        average_score = (
            sum(document_scores)
            / len(document_scores)
        )

    else:

        average_score = 0.0

    return {
        "retrieved_documents": len(results),
        "average_keyword_relevance": round(
            average_score,
            4,
        ),
        "document_scores": [
            round(score, 4)
            for score in document_scores
        ],
    }


# ============================================================
# GROUNDEDNESS
# ============================================================

def evaluate_groundedness(
    answer: str,
    context: str,
) -> Dict[str, Any]:
    """
    Estimate groundedness using vocabulary overlap.

    Higher overlap means more answer terminology
    is present in the retrieved context.

    This is a heuristic and not a factual verifier.
    """

    answer_words = get_words(
        answer
    )

    context_words = get_words(
        context
    )

    if not answer_words:

        score = 0.0

    else:

        overlap = (
            answer_words
            & context_words
        )

        score = (
            len(overlap)
            / len(answer_words)
        )

    return {
        "groundedness_score": round(
            score,
            4,
        ),
        "answer_words": len(
            answer_words
        ),
    }


# ============================================================
# SOURCE COVERAGE
# ============================================================

def evaluate_source_coverage(
    answer: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Estimate explicit source/page citation coverage.

    This checks whether the answer mentions source
    filenames or page numbers.
    """

    answer_lower = answer.lower()

    cited_sources = 0

    source_details = []

    for source in sources:

        filename = source.get(
            "source",
            "",
        )

        page = str(
            source.get(
                "page",
                "",
            )
        )

        filename_present = (
            filename.lower()
            in answer_lower
        )

        page_present = (
            f"page {page}"
            in answer_lower
        )

        cited = (
            filename_present
            or page_present
        )

        if cited:

            cited_sources += 1

        source_details.append(
            {
                "source": filename,
                "page": page,
                "cited": cited,
            }
        )

    if sources:

        coverage = (
            cited_sources
            / len(sources)
        )

    else:

        coverage = 0.0

    return {
        "source_coverage": round(
            coverage,
            4,
        ),
        "cited_sources": cited_sources,
        "total_sources": len(sources),
        "details": source_details,
    }


# ============================================================
# ANSWER STATISTICS
# ============================================================

def answer_statistics(
    answer: str,
) -> Dict[str, Any]:
    """Calculate basic answer statistics."""

    words = answer.split()

    sentences = re.split(
        r"[.!?]+",
        answer,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return {
        "characters": len(answer),
        "words": len(words),
        "sentences": len(sentences),
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    answer: str,
    context: str,
    sources: List[Dict[str, Any]],
    latency: float,
) -> Dict[str, Any]:
    """Evaluate a single model."""

    groundedness = evaluate_groundedness(
        answer,
        context,
    )

    source_coverage = evaluate_source_coverage(
        answer,
        sources,
    )

    statistics = answer_statistics(
        answer,
    )

    return {
        "model": model_name,
        "latency_seconds": round(
            latency,
            3,
        ),
        "groundedness": groundedness,
        "source_coverage": source_coverage,
        "answer_statistics": statistics,
    }


# ============================================================
# PRINT MODEL EVALUATION
# ============================================================

def print_model_evaluation(
    evaluation: Dict[str, Any],
) -> None:
    """Display evaluation metrics for a model."""

    print(
        f"\n{evaluation['model']}"
    )

    print("-" * 70)

    print(
        f"Generation latency: "
        f"{evaluation['latency_seconds']:.3f} sec"
    )

    print(
        f"Groundedness: "
        f"{evaluation['groundedness']['groundedness_score']:.3f}"
    )

    print(
        f"Source coverage: "
        f"{evaluation['source_coverage']['source_coverage']:.3f}"
    )

    print(
        f"Answer words: "
        f"{evaluation['answer_statistics']['words']}"
    )

    print(
        f"Answer characters: "
        f"{evaluation['answer_statistics']['characters']}"
    )


# ============================================================
# COMPARISON
# ============================================================

def print_comparison(
    evaluation: Dict[str, Any],
) -> None:
    """Display a clean model comparison."""

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    gpt4all = evaluation[
        "gpt4all"
    ]

    gemini = evaluation[
        "gemini"
    ]

    print()

    print(
        f"{'Metric':<32}"
        f"{'GPT4All':>15}"
        f"{'Gemini':>15}"
    )

    print("-" * 70)

    print(
        f"{'Generation latency (sec)':<32}"
        f"{gpt4all['latency_seconds']:>15.3f}"
        f"{gemini['latency_seconds']:>15.3f}"
        if gemini
        else
        f"{'Generation latency (sec)':<32}"
        f"{gpt4all['latency_seconds']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'Groundedness':<32}"
        f"{gpt4all['groundedness']['groundedness_score']:>15.3f}"
        f"{gemini['groundedness']['groundedness_score']:>15.3f}"
        if gemini
        else
        f"{'Groundedness':<32}"
        f"{gpt4all['groundedness']['groundedness_score']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'Source coverage':<32}"
        f"{gpt4all['source_coverage']['source_coverage']:>15.3f}"
        f"{gemini['source_coverage']['source_coverage']:>15.3f}"
        if gemini
        else
        f"{'Source coverage':<32}"
        f"{gpt4all['source_coverage']['source_coverage']:>15.3f}"
        f"{'N/A':>15}"
    )

    print(
        f"{'Answer words':<32}"
        f"{gpt4all['answer_statistics']['words']:>15}"
        f"{gemini['answer_statistics']['words']:>15}"
        if gemini
        else
        f"{'Answer words':<32}"
        f"{gpt4all['answer_statistics']['words']:>15}"
        f"{'N/A':>15}"
    )

    print(
        f"{'Answer characters':<32}"
        f"{gpt4all['answer_statistics']['characters']:>15}"
        f"{gemini['answer_statistics']['characters']:>15}"
        if gemini
        else
        f"{'Answer characters':<32}"
        f"{gpt4all['answer_statistics']['characters']:>15}"
        f"{'N/A':>15}"
    )

    print("-" * 70)

    print(
        f"{'Retrieval latency (sec)':<32}"
        f"{evaluation['retrieval_latency_seconds']:>15.3f}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("RAG EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create pipeline
    # --------------------------------------------------------

    pipeline = DualRAGPipeline(
        top_k=TOP_K,
        enable_gemini=True,
    )

    question = (
        "What are Caterpillar's major business risks?"
    )

    print("\nQUESTION")
    print("=" * 70)
    print(question)

    # --------------------------------------------------------
    # Retrieval timing
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    retrieved_documents = (
        pipeline.retriever.retrieve(
            question
        )
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"\nInitial retrieval completed in "
        f"{retrieval_time:.3f} sec"
    )

    # --------------------------------------------------------
    # Run complete dual RAG
    # --------------------------------------------------------

    pipeline_start = time.perf_counter()

    result = pipeline.ask(
        question
    )

    total_pipeline_time = (
        time.perf_counter()
        - pipeline_start
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The current dual_rag_pipeline returns the answers
    # but does not expose individual generation timings.
    #
    # We therefore do not invent timings here.
    # --------------------------------------------------------

    gpt4all_time = 0.0
    gemini_time = 0.0

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = {
        "question": question,

        "retrieval": evaluate_retrieval(
            question,
            result["retrieved_documents"],
        ),

        "gpt4all": evaluate_model(
            model_name="GPT4All / Qwen2-1.5B",
            answer=result["gpt4all_answer"],
            context=result["context"],
            sources=result["sources"],
            latency=gpt4all_time,
        ),

        "gemini": (
            evaluate_model(
                model_name="Gemini",
                answer=result["gemini_answer"],
                context=result["context"],
                sources=result["sources"],
                latency=gemini_time,
            )
            if result.get("gemini_answer")
            else None
        ),

        "retrieval_latency_seconds": (
            retrieval_time
        ),

        "total_pipeline_seconds": (
            total_pipeline_time
        ),
    }

    # --------------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    retrieval = evaluation[
        "retrieval"
    ]

    print(
        f"Documents retrieved: "
        f"{retrieval['retrieved_documents']}"
    )

    print(
        f"Average keyword relevance: "
        f"{retrieval['average_keyword_relevance']:.3f}"
    )

    print(
        f"Retrieval latency: "
        f"{evaluation['retrieval_latency_seconds']:.3f} sec"
    )

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print_model_evaluation(
        evaluation["gpt4all"]
    )

    if evaluation["gemini"]:

        print_model_evaluation(
            evaluation["gemini"]
        )

    else:

        print(
            "\nGemini evaluation unavailable."
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print_comparison(
        evaluation
    )

    print(
        f"\nTotal pipeline execution: "
        f"{evaluation['total_pipeline_seconds']:.3f} sec"
    )

    print(
        "\nRAG evaluation completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
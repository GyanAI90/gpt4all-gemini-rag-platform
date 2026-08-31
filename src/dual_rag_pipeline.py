"""
Dual-Model RAG Pipeline
=======================

Shared retrieval + context is passed to:

    1. GPT4All / Qwen2-1.5B
    2. Gemini 3.6 Flash

The pipeline records timing for every major stage.
"""

import time
from typing import Dict, Any, List

from retriever import DocumentRetriever
from gpt4all_llm import GPT4AllLLM
from gemini_llm import GeminiLLM


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

MAX_CONTEXT_CHARS = 6000


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    results: List[Dict[str, Any]],
) -> str:
    """
    Convert retrieved documents into a shared context.
    """

    context_parts = []

    current_length = 0

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "unknown",
        )

        content = result.get(
            "content",
            "",
        ).strip()

        block = (
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_id}\n"
            f"Content:\n"
            f"{content}\n"
        )

        if (
            current_length
            + len(block)
            > MAX_CONTEXT_CHARS
        ):
            break

        context_parts.append(
            block
        )

        current_length += len(block)

    return "\n".join(
        context_parts
    )


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded RAG prompt.
    """

    return f"""
You are a document question-answering assistant.

Answer the QUESTION using ONLY the information in the
provided CONTEXT.

RULES:

1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context contains relevant information, answer the question.
5. If the answer genuinely cannot be found, say:
   "The answer was not found in the provided documents."
6. Keep the answer concise and factual.
7. Use numbered points when appropriate.
8. Do not repeat the question.
9. When possible, mention the relevant document and page number
   supporting important claims.

CONTEXT
=======

{context}

END CONTEXT

QUESTION
========

{question}

ANSWER
======
""".strip()


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract unique document/page combinations.
    """

    sources = []

    seen = set()

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "unknown",
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            }
        )

    return sources


# ============================================================
# DUAL RAG PIPELINE
# ============================================================

class DualRAGPipeline:
    """
    Runs the same retrieved context through GPT4All
    and Gemini.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        enable_gemini: bool = True,
    ):

        print("=" * 70)
        print("DUAL-MODEL RAG PIPELINE")
        print("=" * 70)

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        print("\nLoading retriever...")

        self.retriever = DocumentRetriever(
            top_k=top_k
        )

        # ----------------------------------------------------
        # GPT4All
        # ----------------------------------------------------

        print("\nLoading GPT4All...")

        self.gpt4all = GPT4AllLLM()

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        self.gemini = None

        if enable_gemini:

            print("\nLoading Gemini...")

            try:

                self.gemini = GeminiLLM()

            except Exception as error:

                print(
                    "\nGemini initialization failed."
                )

                print(
                    f"Reason: {error}"
                )

                print(
                    "GPT4All will remain available."
                )

        print(
            "\nDual-model RAG pipeline ready."
        )

        print("=" * 70)

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Execute the complete dual-model RAG pipeline.

        Returns answers, sources, retrieved documents,
        shared context, and detailed timing information.
        """

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        pipeline_start = time.perf_counter()

        # ----------------------------------------------------
        # 1. Retrieval
        # ----------------------------------------------------

        print(
            "\nRetrieving relevant documents..."
        )

        retrieval_start = (
            time.perf_counter()
        )

        results = self.retriever.retrieve(
            question
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # ----------------------------------------------------
        # 2. Context construction
        # ----------------------------------------------------

        context_start = (
            time.perf_counter()
        )

        context = build_context(
            results
        )

        context_time = (
            time.perf_counter()
            - context_start
        )

        # ----------------------------------------------------
        # 3. Prompt construction
        # ----------------------------------------------------

        prompt_start = (
            time.perf_counter()
        )

        prompt = build_prompt(
            question,
            context,
        )

        prompt_time = (
            time.perf_counter()
            - prompt_start
        )

        # ----------------------------------------------------
        # 4. GPT4All generation
        # ----------------------------------------------------

        print(
            "\nGenerating GPT4All answer..."
        )

        gpt4all_start = (
            time.perf_counter()
        )

        gpt4all_answer = (
            self.gpt4all.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.0,
            )
        )

        gpt4all_time = (
            time.perf_counter()
            - gpt4all_start
        )

        # ----------------------------------------------------
        # 5. Gemini generation
        # ----------------------------------------------------

        gemini_answer = None

        gemini_error = None

        gemini_time = None

        if self.gemini is not None:

            print(
                "\nGenerating Gemini answer..."
            )

            gemini_start = (
                time.perf_counter()
            )

            try:

                gemini_answer = (
                    self.gemini.generate(
                        prompt=prompt,
                    )
                )

            except Exception as error:

                gemini_error = str(
                    error
                )

                print(
                    "\nGemini request failed."
                )

                print(
                    f"Reason: {gemini_error}"
                )

            gemini_time = (
                time.perf_counter()
                - gemini_start
            )

        # ----------------------------------------------------
        # 6. Source extraction
        # ----------------------------------------------------

        source_start = (
            time.perf_counter()
        )

        sources = extract_sources(
            results
        )

        source_time = (
            time.perf_counter()
            - source_start
        )

        # ----------------------------------------------------
        # Total time
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        timing = {
            "retrieval_seconds": round(
                retrieval_time,
                4,
            ),

            "context_seconds": round(
                context_time,
                4,
            ),

            "prompt_seconds": round(
                prompt_time,
                4,
            ),

            "gpt4all_seconds": round(
                gpt4all_time,
                4,
            ),

            "gemini_seconds": (
                round(
                    gemini_time,
                    4,
                )
                if gemini_time is not None
                else None
            ),

            "source_seconds": round(
                source_time,
                4,
            ),

            "total_seconds": round(
                total_time,
                4,
            ),
        }

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "question": question,

            "context": context,

            "retrieved_documents": results,

            "sources": sources,

            "gpt4all_answer": (
                gpt4all_answer
            ),

            "gemini_answer": (
                gemini_answer
            ),

            "gemini_error": (
                gemini_error
            ),

            "timing": timing,
        }


# ============================================================
# DISPLAY
# ============================================================

def print_result(
    result: Dict[str, Any],
) -> None:
    """
    Display the dual-model result.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "GPT4ALL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result["gpt4all_answer"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GEMINI ANSWER"
    )

    print(
        "=" * 70
    )

    if result["gemini_answer"]:

        print(
            result["gemini_answer"]
        )

    else:

        print(
            "Gemini answer unavailable."
        )

        if result["gemini_error"]:

            print(
                f"\nReason: "
                f"{result['gemini_error']}"
            )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE TIMING"
    )

    print(
        "=" * 70
    )

    timing = result[
        "timing"
    ]

    for name, value in timing.items():

        if value is None:

            print(
                f"{name:<25} N/A"
            )

        else:

            print(
                f"{name:<25} "
                f"{value:.4f} sec"
            )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SHARED SOURCES"
    )

    print(
        "=" * 70
    )

    for source in result[
        "sources"
    ]:

        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GPT4ALL + GEMINI RAG PLATFORM"
    )
    print(
        "DUAL-MODEL RAG TEST"
    )
    print("=" * 70)

    pipeline = DualRAGPipeline(
        top_k=5,
        enable_gemini=True,
    )

    question = (
        "What are Caterpillar's major business risks?"
    )

    print(
        "\nUSER QUESTION"
    )

    print(
        "=" * 70
    )

    print(
        question
    )

    result = pipeline.ask(
        question
    )

    print_result(
        result
    )

    print(
        "\nDual-model RAG test completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
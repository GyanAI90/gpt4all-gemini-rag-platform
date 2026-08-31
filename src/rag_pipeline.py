"""
GPT4All RAG Pipeline
====================

Retrieval Augmented Generation using:

    ChromaDB
        +
    Sentence Transformers
        +
    GPT4All / Qwen2-1.5B-Instruct

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from typing import List, Dict, Any

from retriever import DocumentRetriever
from gpt4all_llm import GPT4AllLLM


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

MAX_CONTEXT_CHARS = 5000


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    results: List[Dict[str, Any]],
) -> str:
    """
    Build a clean context string from retrieved documents.
    """

    context_parts = []

    current_length = 0

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        content = result["content"].strip()

        source_block = (
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{content}\n"
        )

        if (
            current_length
            + len(source_block)
            > MAX_CONTEXT_CHARS
        ):
            break

        context_parts.append(
            source_block
        )

        current_length += len(source_block)

    return "\n".join(context_parts)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded prompt optimized for Qwen2-1.5B.
    """

    return f"""
You are a document-based question answering assistant.

You have been given CONTEXT retrieved from documents.

Your job is to answer the QUESTION using the CONTEXT.

STRICT RULES:

- Use only information present in the CONTEXT.
- Do not use outside knowledge.
- Do not invent facts.
- If the CONTEXT contains information that answers the question,
  provide the answer.
- Only say "The answer was not found in the provided documents."
  if the CONTEXT genuinely contains no information relevant
  to the question.
- Do not include the "answer not found" statement after giving
  a valid answer.
- Keep the answer concise and factual.
- Use numbered points when appropriate.

CONTEXT:
========

{context}

END CONTEXT.

QUESTION:
=========

{question}

ANSWER:
=======
""".strip()


# ============================================================
# SOURCE INFORMATION
# ============================================================

def extract_sources(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract source metadata for citation display.
    """

    sources = []

    seen = set()

    for result in results:

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
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
                "chunk_id": metadata.get(
                    "chunk_id",
                    "unknown",
                ),
            }
        )

    return sources


# ============================================================
# RAG PIPELINE
# ============================================================

class GPT4AllRAGPipeline:
    """
    Complete local RAG pipeline.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
    ):

        print("=" * 70)
        print("GPT4ALL RAG PIPELINE")
        print("=" * 70)

        print("\nLoading retriever...")

        self.retriever = DocumentRetriever(
            top_k=top_k
        )

        print("\nLoading GPT4All...")

        self.llm = GPT4AllLLM()

        print("\nRAG pipeline ready.")

        print("=" * 70)

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # ----------------------------------------------------
        # 1. Retrieve
        # ----------------------------------------------------

        results = self.retriever.retrieve(
            question
        )

        # ----------------------------------------------------
        # 2. Build context
        # ----------------------------------------------------

        context = build_context(
            results
        )

        # ----------------------------------------------------
        # 3. Build prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            question,
            context,
        )

        # ----------------------------------------------------
        # 4. Generate answer
        # ----------------------------------------------------

        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )

        # ----------------------------------------------------
        # 5. Extract sources
        # ----------------------------------------------------

        sources = extract_sources(
            results
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_documents": results,
        }


# ============================================================
# DISPLAY
# ============================================================

def print_rag_result(
    result: Dict[str, Any],
) -> None:
    """
    Display the RAG result.
    """

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print("\n" + result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )

    print("=" * 70)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("END-TO-END GPT4ALL RAG TEST")
    print("=" * 70)

    pipeline = GPT4AllRAGPipeline(
        top_k=5
    )

    question = (
        "What are Caterpillar's major business risks?"
    )

    print("\nUSER QUESTION")
    print("=" * 70)

    print(question)

    print("\nRunning GPT4All RAG pipeline...")

    result = pipeline.ask(
        question
    )

    print_rag_result(
        result
    )

    print(
        "\nGPT4All RAG pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
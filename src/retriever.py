"""
Retriever
=========

Retrieves the most relevant document chunks from ChromaDB.

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from pathlib import Path
from typing import List, Dict, Any

from embeddings import EmbeddingModel
from vector_store import ChromaVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# RETRIEVER
# ============================================================

class DocumentRetriever:
    """Semantic retriever backed by ChromaDB."""

    def __init__(
        self,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k

        print("=" * 70)
        print("DOCUMENT RETRIEVER")
        print("=" * 70)

        print("Loading embedding model...")

        self.embedding_model = EmbeddingModel()

        print("Loading ChromaDB vector store...")

        self.vector_store = ChromaVectorStore()

        print(
            f"Retriever configured with top_k={self.top_k}"
        )

        print("=" * 70)

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks.

        Args:
            query: User question.

        Returns:
            List of dictionaries containing content,
            metadata and similarity information.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        results = self.vector_store.search(
            query=query,
            embedding_model=self.embedding_model,
            top_k=self.top_k,
        )

        retrieved_documents = []

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        for index, content in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            # ChromaDB distance is lower for more
            # similar results. This simple conversion
            # is useful for display/diagnostics.
            similarity = (
                1.0 / (1.0 + distance)
                if distance is not None
                else None
            )

            retrieved_documents.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return retrieved_documents


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    query: str,
    results: List[Dict[str, Any]],
) -> None:
    """Display retrieval results."""

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuery:")
    print(query)

    print(
        f"\nRetrieved documents: "
        f"{len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print("\n" + "-" * 70)

        print(f"RESULT #{index}")

        print(
            f"Document: "
            f"{metadata.get('source', 'unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'unknown')}"
        )

        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id', 'unknown')}"
        )

        if result["distance"] is not None:
            print(
                f"Chroma distance: "
                f"{result['distance']:.4f}"
            )

        if result["similarity"] is not None:
            print(
                f"Similarity score: "
                f"{result['similarity']:.4f}"
            )

        print("\nContent:")
        print(
            result["content"][:1000]
        )

    print("\n" + "=" * 70)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("RETRIEVER TEST")
    print("=" * 70)

    retriever = DocumentRetriever(
        top_k=5
    )

    query = (
        "What are Caterpillar's major business risks?"
    )

    results = retriever.retrieve(query)

    print_results(
        query,
        results,
    )

    print(
        "\nRetriever test completed successfully."
    )


if __name__ == "__main__":
    main()
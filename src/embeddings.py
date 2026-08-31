"""
Embedding Model
===============

Creates semantic embeddings for RAG documents.

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

class EmbeddingModel:
    """Wrapper around the Sentence Transformers model."""

    def __init__(self):
        print("=" * 70)
        print("EMBEDDING MODEL")
        print("=" * 70)

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.dimension = (
            self.model.get_embedding_dimension()
        )

        print("Embedding model loaded successfully.")
        print(
            f"Embedding dimensions: {self.dimension}"
        )

        print("=" * 70)

    def embed_text(self, text: str):
        """Generate an embedding for one text."""

        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        return self.model.encode(
            text,
            normalize_embeddings=True,
        )

    def embed_documents(self, documents):
        """Generate embeddings for multiple documents."""

        texts = [
            document.page_content
            for document in documents
        ]

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


# ============================================================
# TEST
# ============================================================

def main():

    embedding_model = EmbeddingModel()

    test_text = (
        "Retrieval Augmented Generation combines "
        "document retrieval with large language models."
    )

    print("\nTest text:")
    print(test_text)

    embedding = embedding_model.embed_text(
        test_text
    )

    print(
        f"\nEmbedding dimensions: "
        f"{len(embedding)}"
    )

    print("\nFirst 10 values:")

    print(embedding[:10].tolist())

    print("\n" + "=" * 70)
    print("Embedding test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
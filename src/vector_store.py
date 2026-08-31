"""
ChromaDB Vector Store
=====================

Creates and loads the local ChromaDB vector database.

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from pathlib import Path
from typing import List

import chromadb
from langchain_core.documents import Document

from chunking import create_chunks
from document_loader import load_all_pdfs
from embeddings import EmbeddingModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "gpt4all_gemini_documents"


# ============================================================
# VECTOR STORE
# ============================================================

class ChromaVectorStore:
    """Local ChromaDB vector store."""

    def __init__(self):

        print("=" * 70)
        print("CHROMADB VECTOR STORE")
        print("=" * 70)

        print(f"Database directory:")
        print(CHROMA_DB_DIR)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": (
                    "PDF document embeddings for "
                    "GPT4All and Gemini RAG"
                )
            },
        )

        print(
            f"Collection: {COLLECTION_NAME}"
        )

        print(
            f"Existing documents: "
            f"{self.collection.count()}"
        )

        print("=" * 70)

    def add_documents(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
    ):
        """
        Add document chunks and embeddings to ChromaDB.
        """

        if not documents:
            raise ValueError(
                "No documents supplied."
            )

        print(
            f"\nCreating embeddings for "
            f"{len(documents)} chunks..."
        )

        embeddings = embedding_model.embed_documents(
            documents
        )

        ids = []
        texts = []
        metadatas = []

        for index, document in enumerate(documents):

            chunk_id = document.metadata.get(
                "chunk_id",
                f"chunk_{index:06d}",
            )

            ids.append(chunk_id)

            texts.append(
                document.page_content
            )

            # Chroma metadata must contain simple values.
            metadata = {
                "source": str(
                    document.metadata.get(
                        "source",
                        "unknown",
                    )
                ),
                "page": int(
                    document.metadata.get(
                        "page",
                        0,
                    )
                ),
                "file_path": str(
                    document.metadata.get(
                        "file_path",
                        "",
                    )
                ),
                "chunk_id": str(chunk_id),
            }

            metadatas.append(metadata)

        print("Adding vectors to ChromaDB...")

        # ChromaDB has practical batch-size limitations,
        # so insert in manageable batches.
        batch_size = 100

        for start in range(
            0,
            len(documents),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(documents),
            )

            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=[
                    embedding.tolist()
                    for embedding in embeddings[start:end]
                ],
                metadatas=metadatas[start:end],
            )

            print(
                f"  Added {end}/{len(documents)}"
            )

        print(
            "\nChromaDB insertion completed."
        )

        print(
            f"Total vectors: "
            f"{self.collection.count()}"
        )

    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
    ):
        """
        Search ChromaDB using semantic similarity.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = (
            embedding_model.embed_text(query)
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("VECTOR DATABASE CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------------

    print("\n[1/4] Loading PDF documents...")

    documents = load_all_pdfs()

    print(
        f"Loaded {len(documents)} pages."
    )

    # --------------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------------

    print("\n[2/4] Creating document chunks...")

    chunks = create_chunks(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------------

    print("\n[3/4] Creating embedding model...")

    embedding_model = EmbeddingModel()

    # --------------------------------------------------------
    # 4. Create vector store
    # --------------------------------------------------------

    print("\n[4/4] Creating ChromaDB vector store...")

    vector_store = ChromaVectorStore()

    vector_store.add_documents(
        chunks,
        embedding_model,
    )

    # --------------------------------------------------------
    # Similarity search test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIMILARITY SEARCH TEST")
    print("=" * 70)

    query = (
        "What are Caterpillar's major business risks?"
    )

    print(f"\nQuery: {query}")

    results = vector_store.search(
        query=query,
        embedding_model=embedding_model,
        top_k=3,
    )

    print(
        f"\nRetrieved "
        f"{len(results['documents'][0])} documents."
    )

    for index, document in enumerate(
        results["documents"][0],
        start=1,
    ):

        metadata = results["metadatas"][0][
            index - 1
        ]

        distance = results["distances"][0][
            index - 1
        ]

        print("\n" + "-" * 60)

        print(f"Result #{index}")

        print(
            f"Document: "
            f"{metadata['source']}"
        )

        print(
            f"Page: "
            f"{metadata['page']}"
        )

        print(
            f"Chunk ID: "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Chroma distance: "
            f"{distance:.4f}"
        )

        print("\nContent preview:")

        print(
            document[:800]
        )

    print("\n" + "=" * 70)
    print("VECTOR DATABASE SETUP COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
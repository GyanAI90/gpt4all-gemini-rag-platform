"""
Document Chunking
=================

Splits loaded PDF documents into overlapping chunks
while preserving document metadata.

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_all_pdfs


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    documents: List[Document],
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents: Loaded PDF documents.

    Returns:
        List of document chunks.
    """

    print("\nCreating document chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # Add a unique chunk ID while preserving existing metadata.
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = (
            f"chunk_{index:06d}"
        )

    return chunks


# ============================================================
# STATISTICS
# ============================================================

def print_chunk_summary(
    chunks: List[Document],
) -> None:
    """
    Print chunking statistics.
    """

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    if chunks:

        average_size = (
            sum(len(chunk.page_content) for chunk in chunks)
            / len(chunks)
        )

        print(
            f"Average chunk size: "
            f"{average_size:.2f} characters"
        )

    # Count chunks by source.
    chunks_by_source = {}

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown",
        )

        chunks_by_source[source] = (
            chunks_by_source.get(source, 0) + 1
        )

    print("\nChunks by document:")

    for source, count in chunks_by_source.items():

        print(f"  - {source}: {count}")

    print("=" * 70)


# ============================================================
# SAMPLE
# ============================================================

def print_sample(
    chunks: List[Document],
) -> None:
    """
    Display a sample chunk.
    """

    if not chunks:
        print("\nNo chunks available.")
        return

    chunk = chunks[0]

    print("\n" + "=" * 70)
    print("SAMPLE CHUNK")
    print("=" * 70)

    print(
        f"Chunk ID: "
        f"{chunk.metadata.get('chunk_id')}"
    )

    print(
        f"Document: "
        f"{chunk.metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{chunk.metadata.get('page')}"
    )

    print(
        f"Chunk size: "
        f"{len(chunk.page_content)} characters"
    )

    print("\nContent:")
    print(chunk.page_content)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("DOCUMENT CHUNKING")
    print("=" * 70)

    print("\nLoading PDF documents...")

    documents = load_all_pdfs()

    print(f"\nLoaded {len(documents)} pages.")

    chunks = create_chunks(documents)

    print_chunk_summary(chunks)

    print_sample(chunks)

    print("\nDocument chunking completed successfully.")


if __name__ == "__main__":
    main()
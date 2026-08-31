"""
Document Loader
================

Loads PDF documents from the data/pdfs directory.

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdfs"


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF file.

    Each PDF page becomes one LangChain Document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects.
    """

    print(f"Loading: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "page": page_number,
            },
        )

        documents.append(document)

    print(f"  Pages with text: {len(documents)}")

    return documents


def load_all_pdfs() -> List[Document]:
    """
    Load all PDF documents from the PDF directory.

    Returns:
        Combined list of documents from all PDFs.
    """

    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory not found:\n{PDF_DIRECTORY}"
        )

    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in:\n{PDF_DIRECTORY}"
        )

    all_documents = []

    print("=" * 70)
    print("GPT4ALL + GEMINI RAG PLATFORM")
    print("DOCUMENT LOADING")
    print("=" * 70)

    print(f"\nPDF directory:")
    print(PDF_DIRECTORY)

    print(f"\nPDF files found: {len(pdf_files)}\n")

    for pdf_file in pdf_files:

        documents = load_pdf(pdf_file)

        all_documents.extend(documents)

    return all_documents


# ============================================================
# SUMMARY
# ============================================================

def print_summary(documents: List[Document]) -> None:
    """
    Print document loading statistics.
    """

    print("\n" + "=" * 70)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 70)

    print(f"Total pages loaded: {len(documents)}")

    documents_by_source = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        documents_by_source[source] = (
            documents_by_source.get(source, 0) + 1
        )

    print("\nDocuments:")

    for source, count in documents_by_source.items():

        print(f"  - {source}: {count} pages")

    print("=" * 70)


# ============================================================
# SAMPLE DOCUMENT
# ============================================================

def print_sample(documents: List[Document]) -> None:
    """
    Display a sample loaded document.
    """

    if not documents:
        print("\nNo documents available.")
        return

    document = documents[0]

    print("\n" + "=" * 70)
    print("FIRST DOCUMENT SAMPLE")
    print("=" * 70)

    print(
        f"Document: "
        f"{document.metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{document.metadata.get('page')}"
    )

    print("\nContent preview:")

    preview = document.page_content[:1000]

    print(preview)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    documents = load_all_pdfs()

    print_summary(documents)

    print_sample(documents)

    print("\nDocument loading completed successfully.")


if __name__ == "__main__":
    main()
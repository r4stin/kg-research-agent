# src/tools/pdf_ingest.py
from pathlib import Path
from typing import List, Dict
import uuid

import chromadb
from pypdf import PdfReader

from src.config import PDF_STORAGE, CHROMA_DB_PATH
from src.embeddings import GeminiEmbeddingFunction


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Very simple character-based chunking with overlap.
    Rough heuristic: ~4 chars ≈ 1 token, so 1200 chars ≈ 300 tokens.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def build_or_update_vector_store():
    """
    Walk over PDF_STORAGE, ingest all PDFs, and store chunks in Chroma.
    This is idempotent-ish: we generate ids that include the filename
    so you can later detect duplicates if you want.
    """
    pdf_dir = Path(PDF_STORAGE)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}. Add at least one and rerun.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    # Set up Chroma persistent client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = GeminiEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=embedding_fn,
    )

    all_documents: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    for pdf_path in pdf_files:
        paper_id = pdf_path.stem  # filename without extension
        print(f"Processing {pdf_path.name} (paper_id={paper_id})")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            chunk_id = f"{paper_id}::chunk-{idx:04d}"

            all_documents.append(chunk)
            all_metadatas.append(
                {
                    "paper_id": paper_id,
                    "chunk_index": idx,
                    "source": str(pdf_path.name),
                }
            )
            all_ids.append(chunk_id)

    if not all_documents:
        print("No non-empty chunks to add.")
        return

    print(f"Adding {len(all_documents)} chunks to Chroma collection 'research_papers'...")

    collection.add(
        ids=all_ids,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    build_or_update_vector_store()

# src/tools/vector_search.py
from typing import List, Dict

import chromadb

from src.config import CHROMA_DB_PATH
from src.embeddings import GeminiEmbeddingFunction


def vector_search(query: str, k: int = 5) -> List[Dict]:
    """
    Query the 'research_papers' collection for the k most similar chunks.
    Returns a list of dicts: {text, paper_id, chunk_index, source, distance}.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = GeminiEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=embedding_fn,
    )

    results = collection.query(
        query_texts=[query],
        n_results=k,
    )

    # Chroma returns lists per query; we have only one query.
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits: List[Dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hit = {
            "text": doc,
            "distance": dist,
            "paper_id": meta.get("paper_id"),
            "chunk_index": meta.get("chunk_index"),
            "source": meta.get("source"),
        }
        hits.append(hit)

    return hits


if __name__ == "__main__":
    # quick manual test
    from pprint import pprint
    q = "What is a major challange in scholarly information retrieval?"
    pprint(vector_search(q, k=3))


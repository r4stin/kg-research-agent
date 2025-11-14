# src/utils/format_hits.py

from typing import List, Dict


def format_hits_for_prompt(hits: List[Dict]) -> str:
    """
    Turn vector_search hits into a readable text block for the LLM.

    Each hit looks like:
      {
        "text": "...",
        "distance": float,
        "paper_id": "...",
        "chunk_index": int,
        "source": "..."
      }
    """
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[HIT {i}] paper_id={hit.get('paper_id')}, "
                     f"chunk_index={hit.get('chunk_index')}, "
                     f"source={hit.get('source')}, "
                     f"distance={hit.get('distance'):.4f}")
        lines.append(hit["text"].strip())
        lines.append("")  # blank line between hits

    return "\n".join(lines)

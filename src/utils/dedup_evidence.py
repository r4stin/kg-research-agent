# src/utils/dedup_evidence.py

from typing import Set, Tuple

from src.models.evidence import EvidenceResponse, EvidenceItem


def deduplicate_evidence(evidence: EvidenceResponse) -> EvidenceResponse:
    """
    Remove duplicate or near-duplicate EvidenceItems.

    Strategy:
    - Use (paper_id, chunk_index, normalized_claim) as key.
    - Keep only the first item per key.
    """
    seen: Set[Tuple[str, int, str]] = set()
    unique_items = []

    for item in evidence.items:
        key = (
            item.paper_id,
            item.chunk_index,
            item.claim.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    return EvidenceResponse(question=evidence.question, items=unique_items)

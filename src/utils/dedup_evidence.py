# src/utils/dedup_evidence_strict.py

from typing import Set, Tuple, List
from hashlib import sha1
from difflib import SequenceMatcher

from src.models.evidence import EvidenceResponse, EvidenceItem


def _normalize_text(text: str) -> str:
    """Lowercase & collapse whitespace."""
    return " ".join(text.lower().split())


def _question_hash(question: str) -> str:
    """
    Short stable hash for the question, so we can use it as an ID/key
    without storing the full question in the identity.
    """
    norm = _normalize_text(question)
    return sha1(norm.encode("utf-8")).hexdigest()[:16]


def _similar(a: str, b: str, threshold: float = 0.9) -> bool:
    """
    Simple similarity using SequenceMatcher.
    threshold=0.9 is quite strict: only very similar paraphrases are merged.
    """
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    score = SequenceMatcher(None, a_norm, b_norm).ratio()
    return score >= threshold


def deduplicate_evidence(
    evidence: EvidenceResponse,
    claim_similarity_threshold: float = 0.9,
) -> EvidenceResponse:
    """
    Strict deduplication:

    - Group items by (paper_id, chunk_index, question_hash).
    - Within each group, collapse near-duplicate claims (based on text similarity).
    - Keep only one EvidenceItem per representative claim.

    This ensures:
      - Re-running the same question on the same paper/chunk does NOT explode entries.
      - Small paraphrases of the same claim are merged.
    """

    q_hash = _question_hash(evidence.question)

    # Group by (paper_id, chunk_index, question_hash)
    grouped: dict[Tuple[str, int, str], List[EvidenceItem]] = {}

    for item in evidence.items:
        key = (item.paper_id, item.chunk_index, q_hash)
        grouped.setdefault(key, []).append(item)

    new_items: List[EvidenceItem] = []

    for (paper_id, chunk_idx, _), items in grouped.items():
        # We'll maintain representative claims for this group
        reps: List[EvidenceItem] = []

        for candidate in items:
            if not reps:
                reps.append(candidate)
                continue

            # Check if this candidate's claim is similar to any existing rep
            is_duplicate = False
            for rep in reps:
                if _similar(candidate.claim, rep.claim, threshold=claim_similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                reps.append(candidate)

        new_items.extend(reps)

    return EvidenceResponse(question=evidence.question, items=new_items)

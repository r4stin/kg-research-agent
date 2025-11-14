# tests/test_dedup_evidence_strict.py

from src.models.evidence import EvidenceItem, EvidenceResponse
from src.utils.dedup_evidence import deduplicate_evidence


def make_item(paper_id, chunk_index, claim, sentence, source="test.pdf"):
    return EvidenceItem(
        claim=claim,
        evidence_sentence=sentence,
        paper_id=paper_id,
        chunk_index=chunk_index,
        source=source,
    )


def test_dedup_same_claim_same_chunk():
    evidence = EvidenceResponse(
        question="What is a major challenge in scholarly information retrieval?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content like figures is a significant challenge.",
                "Some sentence A.",
            ),
            make_item(
                "paper1",
                1,
                "Integrating multimodal content like figures is a significant challenge.",
                "Some sentence B.",
            ),
        ],
    )

    deduped = deduplicate_evidence(evidence)
    assert len(deduped.items) == 1
    assert "multimodal content" in deduped.items[0].claim


def test_dedup_similar_claims_same_chunk():
    evidence = EvidenceResponse(
        question="What is a major challenge in scholarly information retrieval?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content like figures is a significant challenge.",
                "Sentence A.",
            ),
            make_item(
                "paper1",
                1,
                "Integration of multimodal figures is a major challenge.",
                "Sentence B.",
            ),
        ],
    )

    # Here we explicitly want a more aggressive dedup to treat paraphrases
    # as the same claim, so we lower the threshold.
    deduped = deduplicate_evidence(
        evidence,
        claim_similarity_threshold=0.7,
    )

    assert len(deduped.items) == 1


def test_no_dedup_across_different_chunks():
    evidence = EvidenceResponse(
        question="What is a major challenge in scholarly information retrieval?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content is a significant challenge.",
                "Sentence A.",
            ),
            make_item(
                "paper1",
                2,
                "Integrating multimodal content is a significant challenge.",
                "Sentence B.",
            ),
        ],
    )

    deduped = deduplicate_evidence(evidence)
    # Different chunk_index → should not be merged
    assert len(deduped.items) == 2


def test_no_dedup_across_different_questions():
    evidence1 = EvidenceResponse(
        question="What is a major challenge in scholarly information retrieval?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content is a significant challenge.",
                "Sentence A.",
            ),
        ],
    )

    evidence2 = EvidenceResponse(
        question="What is a major challenge in scholarly text mining?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content is a significant challenge.",
                "Sentence A.",
            ),
        ],
    )

    d1 = deduplicate_evidence(evidence1)
    d2 = deduplicate_evidence(evidence2)

    # Each question is deduped separately; both should keep their one item
    assert len(d1.items) == 1
    assert len(d2.items) == 1


def test_dedup_similar_claims_default_is_conservative():
    evidence = EvidenceResponse(
        question="What is a major challenge in scholarly information retrieval?",
        items=[
            make_item(
                "paper1",
                1,
                "Integrating multimodal content like figures is a significant challenge.",
                "Sentence A.",
            ),
            make_item(
                "paper1",
                1,
                "Integration of multimodal figures is a major challenge.",
                "Sentence B.",
            ),
        ],
    )

    deduped = deduplicate_evidence(evidence)  # default threshold=0.9
    # With a strict threshold, paraphrases are not merged.
    assert len(deduped.items) == 2
# src/models/evidence.py

from typing import List
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    claim: str = Field(
        ...,
        description="A concise scientific claim answering (part of) the question.",
    )
    evidence_sentence: str = Field(
        ...,
        description="A single sentence copied or almost exactly copied from the source text that supports the claim.",
    )
    paper_id: str = Field(
        ...,
        description="Identifier of the paper (e.g., filename without extension).",
    )
    chunk_index: int = Field(
        ...,
        description="Index of the chunk where this sentence was found.",
    )
    source: str = Field(
        ...,
        description="Human-readable source, usually the PDF filename.",
    )


class EvidenceResponse(BaseModel):
    question: str
    items: List[EvidenceItem]

# src/models/agent_messages.py
from typing import List
from pydantic import BaseModel


class ResearchQuery(BaseModel):
    question: str


class PlannerTask(BaseModel):
    task_type: str  # e.g., "retrieval", "evidence", "summary"
    query: str


class RetrievedChunk(BaseModel):
    chunk: str
    paper_id: str
    chunk_index: int
    source: str


class RetrievedContext(BaseModel):
    query: str
    chunks: List[RetrievedChunk]


class EvidenceItem(BaseModel):
    claim: str
    evidence_sentence: str
    paper_id: str
    chunk_index: int
    source: str


class EvidenceBatch(BaseModel):
    question: str
    items: List[EvidenceItem]


class FinalAnswer(BaseModel):
    question: str
    answer: str
    citations: List[EvidenceItem]


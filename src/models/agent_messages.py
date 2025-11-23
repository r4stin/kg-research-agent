# src/models/agent_messages.py
from typing import List
from pydantic import BaseModel

from src.models.evidence import EvidenceItem  # reuse existing model


class ResearchQuery(BaseModel):
    """Top-level user intent passed into the planner."""
    question: str


class PlannerTask(BaseModel):
    """
    A single step the planner wants to execute.

    task_type:
      - "retrieval"  → call retriever_agent
      - "evidence"   → run evidence extraction on given context
      - "answer"     → synthesize final answer
      - (future) "kg_query", "refine", etc.
    """
    task_type: str
    query: str


class RetrievedChunk(BaseModel):
    """One retrieved chunk from the vector store."""
    chunk: str
    paper_id: str
    chunk_index: int
    source: str


class RetrievedContext(BaseModel):
    """What the retriever hands to the evidence agent."""
    query: str
    chunks: List[RetrievedChunk]


class EvidenceBatch(BaseModel):
    """What the evidence agent hands to downstream steps."""
    question: str
    items: List[EvidenceItem]


class FinalAnswer(BaseModel):
    """Final output of the system."""
    question: str
    answer: str
    citations: List[EvidenceItem]

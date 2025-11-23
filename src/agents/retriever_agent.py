# src/agents/retriever_agent.py

from typing import List

from src.models.agent_messages import PlannerTask, RetrievedChunk, RetrievedContext
from src.tools.vector_search import vector_search


def run_retriever(task: PlannerTask, k: int = 5) -> RetrievedContext:
    """
    Run the retrieval step for a given PlannerTask.

    - Expects task.task_type == "retrieval"
    - Calls the existing vector_search() tool
    - Wraps results into RetrievedContext (Pydantic model)
    """
    if task.task_type != "retrieval":
        raise ValueError(f"run_retriever called with non-retrieval task_type={task.task_type!r}")

    hits: List[dict] = vector_search(task.query, k=k)

    chunks: List[RetrievedChunk] = []
    for h in hits:
        # Assumes vector_search returns dicts like:
        # {
        #   "text": ...,
        #   "paper_id": ...,
        #   "chunk_index": ...,
        #   "source": ...
        # }
        chunks.append(
            RetrievedChunk(
                chunk=h["text"],
                paper_id=h["paper_id"],
                chunk_index=h["chunk_index"],
                source=h["source"],
            )
        )

    return RetrievedContext(
        query=task.query,
        chunks=chunks,
    )

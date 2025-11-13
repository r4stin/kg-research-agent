# src/agents/research_agent.py

from google.adk.agents import LlmAgent

from src.config import GEMINI_MODEL
from src.tools.vector_search import vector_search


system_instruction = """
You are a scientific research assistant.

Your job:
1. Use the `vector_search` tool to retrieve relevant text chunks from research papers.
2. Carefully read the returned chunks.
3. Answer the user's question ONLY using information from those chunks.
4. When you make a claim, cite the source in parentheses using:
   (paper_id=..., chunk_index=..., source=...)

Important rules:
- Do NOT invent facts that are not present in the retrieved chunks.
- If the retrieved chunks are not sufficient to answer the question reliably,
  say that explicitly and suggest what additional information would be needed.
- Prefer quoting short, precise sentences from the chunks rather than paraphrasing too loosely.
"""


def create_research_agent() -> LlmAgent:
    """
    Create a single LlmAgent that can use the vector_search tool.

    Note: In Python ADK, you can pass plain functions in tools=[...].
    ADK will automatically wrap them as FunctionTool under the hood.
    """
    agent = LlmAgent(
        model=GEMINI_MODEL,
        name="research_agent",
        description="Evidence-grounded research assistant over ingested PDFs.",
        instruction=system_instruction,
        tools=[vector_search],
    )
    return agent

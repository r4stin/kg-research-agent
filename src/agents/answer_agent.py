# src/agents/answer_agent.py

import asyncio

from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.config import GEMINI_MODEL
from src.models.agent_messages import FinalAnswer, EvidenceBatch
from src.models.evidence import EvidenceItem

system_instruction = """
You are a scientific answering assistant.

You will be given:
- a research question
- a small list of structured evidence items, each with:
  - claim
  - evidence_sentence
  - paper_id
  - chunk_index
  - source (PDF filename)

Your job is to write a clear, concise answer to the question,
STRICTLY based on the provided evidence.

Guidelines:

1. Do NOT invent facts beyond what is in the evidence.
2. If the evidence is insufficient or incomplete, say that explicitly.
3. When you state something, support it with citations using this format:
   [C1], [C2], ...
4. After the main answer, add a short "Evidence" section that lists each citation:

   [C1] <evidence_sentence> (paper_id=..., chunk_index=..., source=...)
   [C2] ...

5. Prefer to:
   - combine related claims,
   - make the logical connection explicit (e.g., "because", "therefore"),
   - avoid over-claiming beyond what the evidence supports.
"""


def create_answer_agent() -> LlmAgent:
    return LlmAgent(
        model=GEMINI_MODEL,
        name="answer_agent",
        description="Composes a natural language answer based on structured evidence items.",
        instruction=system_instruction,
        tools=[],  # no tools, just reasoning over given text
    )


async def _run_answer_agent_async(evidence: EvidenceBatch) -> FinalAnswer:
    """Internal async helper using ADK Runner + session."""

    agent = create_answer_agent()
    app_name = "kg-research-agent-answer"
    user_id = "local_user"
    session_id = "answer-session-1"

    session_service = InMemorySessionService()
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Format evidence items for the prompt
    lines = []
    for idx, item in enumerate(evidence.items, start=1):
        lines.append(
            f"[E{idx}] CLAIM: {item.claim}\n"
            f"    EVIDENCE: {item.evidence_sentence}\n"
            f"    META: paper_id={item.paper_id}, chunk_index={item.chunk_index}, source={item.source}"
        )
    evidence_block = "\n\n".join(lines) if lines else "(no evidence items)"

    prompt = f"""
Question:
{evidence.question}

Evidence items:
{evidence_block}

Write the answer and evidence section as described in your instructions.
"""

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    events = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    )

    final_text = None
    for event in events:
        if event.is_final_response():
            final_text = event.content.parts[0].text

    if not final_text:
        raise RuntimeError("Answer agent returned no final response.")

    # We keep citations as the full evidence items we passed in
    return FinalAnswer(
        question=evidence.question,
        answer=final_text,
        citations=evidence.items,
    )


def run_answer_agent(evidence: EvidenceBatch) -> FinalAnswer:
    """Sync wrapper used by the pipeline."""
    return asyncio.run(_run_answer_agent_async(evidence))

# src/agents/evidence_agent.py

import asyncio
import json

from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.config import GEMINI_MODEL
from src.models.agent_messages import RetrievedContext, EvidenceBatch
from src.models.evidence import EvidenceItem

system_instruction = """
You are an evidence extraction assistant.

Given:
- a research question, and
- a set of retrieved text chunks from scientific papers (with metadata),

you must extract a SMALL number of key claims that answer the question,
and for each claim, provide ONE supporting evidence sentence and full citation.

Return your result as STRICT JSON, with the following structure:

{
  "question": "<the original question>",
  "items": [
    {
      "claim": "<concise claim>",
      "evidence_sentence": "<one supporting sentence from the text>",
      "paper_id": "<paper identifier from metadata>",
      "chunk_index": <chunk index from metadata as integer>,
      "source": "<source filename from metadata>"
    },
    ...
  ]
}

Guidelines:
- Use ONLY the provided chunks. Do NOT invent facts.
- Prefer sentences that clearly, directly support the claim.
- You may output between 1 and 5 items.
- The 'evidence_sentence' should be copied or minimally edited from the chunk text.
- If the chunks do not contain enough information, return an empty 'items' list.
"""


def create_evidence_agent() -> LlmAgent:
    return LlmAgent(
        model=GEMINI_MODEL,
        name="evidence_agent",
        description="Extracts structured evidence (claim + supporting sentence) from retrieved chunks.",
        instruction=system_instruction,
        tools=[],  # no tools; we pass all context in the prompt
    )


async def _run_evidence_agent_async(ctx: RetrievedContext, question: str) -> EvidenceBatch:
    """Internal async helper that uses ADK Runner + session service."""

    agent = create_evidence_agent()
    app_name = "kg-research-agent-evidence"
    user_id = "local_user"
    session_id = "evidence-session-1"

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

    # Build context string from retrieved chunks
    formatted_chunks = "\n\n".join(
        f"[CHUNK] paper_id={c.paper_id}, chunk_index={c.chunk_index}, source={c.source}\n{c.chunk}"
        for c in ctx.chunks
    )

    prompt = f"""
Question: {question}

You are given retrieved chunks of paper text:

{formatted_chunks}

Extract claims and supporting evidence as per your JSON schema.
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

    json_text = None
    for event in events:
        if event.is_final_response():
            # Final LLM response
            json_text = event.content.parts[0].text

    if not json_text:
        raise RuntimeError("Evidence agent returned no final response.")

    # Strip ```json ... ``` if the model wrapped it
    cleaned = json_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Failed to parse evidence JSON: {e}\nRaw: {json_text}")

    items_raw = data.get("items", [])

    items = [
        EvidenceItem(
            claim=it["claim"],
            evidence_sentence=it["evidence_sentence"],
            paper_id=it["paper_id"],
            chunk_index=it["chunk_index"],
            source=it["source"],
        )
        for it in items_raw
    ]

    return EvidenceBatch(
        question=question,
        items=items,
    )


def run_evidence_agent(ctx: RetrievedContext, question: str) -> EvidenceBatch:
    """Sync wrapper so the rest of the code doesn't need to care about asyncio."""
    return asyncio.run(_run_evidence_agent_async(ctx, question))

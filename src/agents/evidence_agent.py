# src/agents/evidence_agent.py

import json
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from src.models.agent_messages import RetrievedContext, EvidenceItem, EvidenceBatch
from src.config import GEMINI_MODEL

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
    agent = LlmAgent(
        model=GEMINI_MODEL,
        name="evidence_agent",
        description="Extracts structured evidence (claim + supporting sentence) from retrieved chunks.",
        instruction=system_instruction,
        tools=[],  # no tools; we pass all context in the prompt
    )
    return agent


def run_evidence_agent(ctx: RetrievedContext, question: str) -> EvidenceBatch:
    """
    Takes retrieved chunks and a research question, sends them to the evidence agent,
    parses JSON response, and returns EvidenceBatch (Pydantic).
    """

    agent = create_evidence_agent()
    runner = Runner(agents=[agent])

    # Build prompt
    formatted_chunks = "\n\n".join(
        f"[CHUNK] paper_id={c.paper_id}, chunk_index={c.chunk_index}\n{c.chunk}"
        for c in ctx.chunks
    )

    prompt = f"""
Question: {question}

You are given retrieved chunks of paper text:

{formatted_chunks}

Extract claims and supporting evidence.
"""

    # Call agent
    result = runner.run_sync(
        agent_name=agent.name,
        message=prompt
    )

    # Extract and parse JSON output
    raw = result.messages[-1].content[0].text  # safe for now
    parsed = json.loads(raw)

    # Convert to typed EvidenceBatch
    items = [
        EvidenceItem(
            claim=i["claim"],
            evidence_sentence=i["evidence_sentence"],
            paper_id=i["paper_id"],
            chunk_index=i["chunk_index"],
            source=i["source"],
        )
        for i in parsed.get("items", [])
    ]

    return EvidenceBatch(
        question=question,
        items=items
    )
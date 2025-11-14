# src/agents/evidence_agent.py

from google.adk.agents import LlmAgent

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

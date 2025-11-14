# src/run_evidence_extraction.py

import asyncio
import json
from typing import List, Dict

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.tools.vector_search import vector_search
from src.utils.format_hits import format_hits_for_prompt
from src.agents.evidence_agent import create_evidence_agent
from src.models.evidence import EvidenceResponse


APP_NAME = "kg-research-agent-evidence-app"
USER_ID = "local_user"
SESSION_ID = "evidence-session-1"


async def main():
    # 1. Get user question
    question = input("Enter your research question: ")

    # 2. Retrieve chunks via vector_search
    hits: List[Dict] = vector_search(question, k=5)
    if not hits:
        print("No hits returned from vector_search. Did you ingest PDFs?")
        return

    context_block = format_hits_for_prompt(hits)

    # 3. Build the prompt content for the evidence agent
    prompt_text = f"""
Research question:
{question}

Retrieved chunks (with metadata):
{context_block}

Now extract structured evidence as JSON according to your instructions.
"""

    # 4. Set up agent, session, and runner
    agent = create_evidence_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt_text)],
    )

    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message,
    )

    final_text = None
    for event in events:
        if event.is_final_response():
            final_text = event.content.parts[0].text

    if not final_text:
        print("No final response from evidence agent.")
        return

    print("\n=== Raw LLM output ===")
    print(final_text)

    # 5. Try to parse as JSON into our Pydantic model
    try:
        # Sometimes the model may wrap JSON in markdown ```json blocks; strip them.
        cleaned = final_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove possible 'json' start
            cleaned = cleaned.replace("json", "", 1).strip()

        data = json.loads(cleaned)
        evidence = EvidenceResponse(**data)

    except Exception as e:
        print("\nCould not parse JSON into EvidenceResponse:", e)
        return

    print("\n=== Parsed EvidenceResponse ===")
    print(f"Question: {evidence.question}")
    for i, item in enumerate(evidence.items, start=1):
        print(f"\n[Item {i}]")
        print("Claim:", item.claim)
        print("Evidence sentence:", item.evidence_sentence)
        print(f"Source: {item.source} (paper_id={item.paper_id}, chunk_index={item.chunk_index})")


if __name__ == "__main__":
    asyncio.run(main())

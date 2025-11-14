# src/pipelines/run_evidence_and_answer.py

import asyncio
import json
from typing import List, Dict

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.tools.vector_search import vector_search
from src.utils.format_hits import format_hits_for_prompt
from src.agents.evidence_agent import create_evidence_agent
from src.agents.answer_agent import create_answer_agent
from src.models.evidence import EvidenceResponse
from src.utils.dedup_evidence import deduplicate_evidence


APP_NAME_EVIDENCE = "kg-research-agent-evidence-app"
APP_NAME_ANSWER = "kg-research-agent-answer-app"
USER_ID = "local_user"


async def main():
    question = input("Enter your research question: ")

    # 1. Retrieve chunks via vector_search
    hits: List[Dict] = vector_search(question, k=5)
    if not hits:
        print("No hits returned from vector_search. Did you ingest PDFs?")
        return

    context_block = format_hits_for_prompt(hits)

    # 2. Run evidence_agent to get structured EvidenceResponse
    evidence_agent = create_evidence_agent()
    evidence_session_service = InMemorySessionService()
    evidence_runner = Runner(
        app_name=APP_NAME_EVIDENCE,
        agent=evidence_agent,
        session_service=evidence_session_service,
    )

    evidence_session_id = "evidence-session-1"
    await evidence_session_service.create_session(
        app_name=APP_NAME_EVIDENCE,
        user_id=USER_ID,
        session_id=evidence_session_id,
    )

    evidence_prompt = f"""
Research question:
{question}

Retrieved chunks (with metadata):
{context_block}

Now extract structured evidence as JSON according to your instructions.
"""

    evidence_message = types.Content(
        role="user",
        parts=[types.Part(text=evidence_prompt)],
    )

    events = evidence_runner.run(
        user_id=USER_ID,
        session_id=evidence_session_id,
        new_message=evidence_message,
    )

    evidence_text = None
    for event in events:
        if event.is_final_response():
            evidence_text = event.content.parts[0].text

    if not evidence_text:
        print("No final response from evidence agent.")
        return

    # Parse into EvidenceResponse
    cleaned = evidence_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        data = json.loads(cleaned)
        evidence = EvidenceResponse(**data)
        evidence = deduplicate_evidence(evidence)
    except Exception as e:
        print("Could not parse evidence JSON:", e)
        print("Raw evidence_text:\n", evidence_text)
        return

    print("\n=== Structured Evidence ===")
    print(f"Question: {evidence.question}")
    for i, item in enumerate(evidence.items, start=1):
        print(f"\n[Item {i}]")
        print("Claim:", item.claim)
        print("Evidence:", item.evidence_sentence)
        print(f"Source: {item.source} (paper_id={item.paper_id}, chunk_index={item.chunk_index})")

    # 3. Run answer_agent to compose final answer based on structured evidence

    answer_agent = create_answer_agent()
    answer_session_service = InMemorySessionService()
    answer_runner = Runner(
        app_name=APP_NAME_ANSWER,
        agent=answer_agent,
        session_service=answer_session_service,
    )

    answer_session_id = "answer-session-1"
    await answer_session_service.create_session(
        app_name=APP_NAME_ANSWER,
        user_id=USER_ID,
        session_id=answer_session_id,
    )

    # Build a compact evidence block for the answer_agent
    evidence_lines = []
    for idx, item in enumerate(evidence.items, start=1):
        evidence_lines.append(
            f"[C{idx}] claim: {item.claim}\n"
            f"    evidence_sentence: {item.evidence_sentence}\n"
            f"    paper_id={item.paper_id}, chunk_index={item.chunk_index}, source={item.source}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    answer_prompt = f"""
Research question:
{evidence.question}

You are provided with the following structured evidence items:

{evidence_block}

Now write an answer following your instructions:
- Answer the question.
- Only use information supported by these evidence items.
- Use [C1], [C2], ... to cite the evidence.
"""

    answer_message = types.Content(
        role="user",
        parts=[types.Part(text=answer_prompt)],
    )

    answer_events = answer_runner.run(
        user_id=USER_ID,
        session_id=answer_session_id,
        new_message=answer_message,
    )

    final_answer = None
    for event in answer_events:
        if event.is_final_response():
            final_answer = event.content.parts[0].text

    print("\n=== Final Answer ===")
    if final_answer:
        print(final_answer)
    else:
        print("No final answer from answer agent.")


if __name__ == "__main__":
    asyncio.run(main())

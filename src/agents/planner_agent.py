# src/agents/planner_agent.py

import json
from typing import List

from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.config import GEMINI_MODEL
from src.models.agent_messages import ResearchQuery, PlannerTask


PLANNER_SYSTEM_PROMPT = """
You are a planning agent that decides how to answer research questions
about scientific papers.

Your job is to produce a small list of tasks that downstream agents will execute.

Available task types:
- "retrieval": retrieve relevant chunks from the vector store.
- "evidence": run evidence extraction over already retrieved context.
- "answer": synthesize the final answer from structured evidence.

Guidelines:
- For most research questions, you will:
  1) call "retrieval" with the original question,
  2) then "evidence" with the same question,
  3) then "answer" with the same question.
- If the question is clearly unanswerable or off-topic, respond with an empty list.
- Do NOT include tools or implementation details, just tasks.

Return STRICTLY valid JSON with this shape:

{
  "tasks": [
    {
      "task_type": "<retrieval|evidence|answer>",
      "query": "<string>"
    },
    ...
  ]
}
"""


def create_planner_agent() -> LlmAgent:
    """Create an ADK LlmAgent that does only planning (no tools)."""
    return LlmAgent(
        model=GEMINI_MODEL,
        name="planner_agent",
        description="Plans which agents should run in which order.",
        instruction=PLANNER_SYSTEM_PROMPT,
        tools=[],  # planner doesn't call tools directly
    )


def plan_question(question: str) -> List[PlannerTask]:
    """
    Convenience helper: run the planner agent once
    and parse the resulting JSON into PlannerTask objects.
    """

    # Wrap question in our Pydantic model (for clarity/type safety)
    rq = ResearchQuery(question=question)

    agent = create_planner_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="kg-research-agent-planner",
        agent=agent,
        session_service=session_service,
    )

    user_id = "local_user"
    session_id = "planner-session-1"

    # create session if needed
    import asyncio
    asyncio.run(
        session_service.create_session(
            app_name="kg-research-agent-planner",
            user_id=user_id,
            session_id=session_id,
        )
    )

    prompt = (
        "You will receive a research question.\n"
        "Decide which tasks to run, following your instructions.\n\n"
        f"Question: {rq.question}\n\n"
        "Return only JSON as specified."
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    events = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    )

    json_text = None
    for event in events:
        if event.is_final_response():
            json_text = event.content.parts[0].text

    if not json_text:
        raise RuntimeError("Planner agent returned no final response.")

    # Handle ```json ... ``` wrappers if present
    cleaned = json_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        data = json.loads(cleaned)
        tasks_raw = data.get("tasks", [])
        tasks = [PlannerTask(**t) for t in tasks_raw]
        return tasks
    except Exception as e:
        raise RuntimeError(f"Failed to parse planner JSON: {e}\nRaw: {json_text}")

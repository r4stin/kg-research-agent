# src/agents/planner_agent.py

import json
from typing import List, Optional

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


def plan_question(question: str, history_context: Optional[str] = None) -> List[PlannerTask]:
    """
    Run the planner agent once and parse the resulting JSON into PlannerTask objects.
    Optionally include short session history as context.
    """

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

    import asyncio
    asyncio.run(
        session_service.create_session(
            app_name="kg-research-agent-planner",
            user_id=user_id,
            session_id=session_id,
        )
    )

    history_block = ""
    if history_context:
        history_block = (
            "Here is the recent conversation history you should consider "
            "when planning (it may contain follow-up questions):\n\n"
            f"{history_context}\n\n"
            "End of history.\n"
        )

    prompt = (
        "You will receive a research question.\n"
        "Decide which tasks to run, following your instructions.\n\n"
        f"{history_block}"
        f"Current question: {rq.question}\n\n"
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

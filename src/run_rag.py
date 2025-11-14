# src/run_rag.py

import asyncio

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.agents.research_agent import create_research_agent


APP_NAME = "kg-research-agent-app"
USER_ID = "local_user"
SESSION_ID = "session-1"


async def main():
    # 1. Create the agent
    agent = create_research_agent()

    # 2. Set up session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    # 3. Create a session (async)
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # 4. Ask the user for a question
    question = input("Enter your research question: ")

    # 5. Wrap the question as a generative-content message
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=question)],
    )

    # 6. Run the agent and process events
    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message,
    )

    print("\n=== Agent events ===")
    final_text = None
    for event in events:
        # You can print all events for debugging:
        # print(event)

        if event.is_final_response():
            # Final LLM response
            final_text = event.content.parts[0].text

    print("\n=== Agent Response ===")
    if final_text:
        print(final_text)
    else:
        print("No final response from agent.")


if __name__ == "__main__":
    asyncio.run(main())

# src/pipelines/run_multi_agent_pipeline.py

from src.agents.planner_agent import plan_question
from src.agents.retriever_agent import run_retriever
from src.agents.evidence_agent import run_evidence_agent
from src.agents.answer_agent import run_answer_agent
from src.models.session_state import SessionState
from src.models.agent_messages import FinalAnswer


def handle_one_turn(question: str, session_state: SessionState) -> FinalAnswer:
    """Run one full planner → retriever → evidence → answer cycle with session memory."""

    # 1) Build history context for the planner (short-term memory)
    history_context = session_state.build_history_context(max_turns=3)

    # 2) Plan with history
    tasks = plan_question(question, history_context=history_context)

    retrieval_task = next((t for t in tasks if t.task_type == "retrieval"), None)
    evidence_task = next((t for t in tasks if t.task_type == "evidence"), None)
    answer_task = next((t for t in tasks if t.task_type == "answer"), None)

    if not retrieval_task or not evidence_task or not answer_task:
        raise RuntimeError(f"Planner did not return the expected sequence. Tasks: {tasks}")

    # 3) Retrieval
    ctx = run_retriever(retrieval_task, k=5)

    # 4) Evidence extraction
    evidence_batch = run_evidence_agent(ctx, question)

    # 5) Final answer
    final: FinalAnswer = run_answer_agent(evidence_batch)

    # 6) Update session memory
    session_state.add_turn(question=question, answer=final.answer)

    return final


def main():
    print("Multi-agent research assistant with session memory.")
    print("Type 'exit' to quit.\n")

    session_state = SessionState()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            final = handle_one_turn(question, session_state)
        except Exception as e:
            print(f"[Error] {e}")
            continue

        print("\nAssistant:\n")
        print(final.answer)
        print("\n---\n")


if __name__ == "__main__":
    main()

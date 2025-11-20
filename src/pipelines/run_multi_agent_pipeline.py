from src.agents.planner_agent import plan_question
from src.agents.retriever_agent import run_retriever
from src.agents.evidence_agent import run_evidence_agent
from src.agents.answer_agent import run_answer_agent
from src.models.agent_messages import FinalAnswer


def main():
    question = input("Enter a research question: ")

    # 1) Planning
    tasks = plan_question(question)

    retrieval_task = next((t for t in tasks if t.task_type == "retrieval"), None)
    evidence_task = next((t for t in tasks if t.task_type == "evidence"), None)
    answer_task = next((t for t in tasks if t.task_type == "answer"), None)

    if not retrieval_task or not evidence_task or not answer_task:
        print("Planner did not return the expected retrieval/evidence/answer sequence.")
        print("Tasks:", tasks)
        return

    # 2) Retrieval
    ctx = run_retriever(retrieval_task, k=5)

    # 3) Evidence extraction
    evidence_batch = run_evidence_agent(ctx, question)

    # 4) Final answer
    final: FinalAnswer = run_answer_agent(evidence_batch)

    print("\n=== Final Answer ===\n")
    print(final.answer)
    print("\n=== Citations (from structured evidence) ===")
    for i, item in enumerate(final.citations, start=1):
        print(f"[C{i}] {item.evidence_sentence} "
              f"(paper_id={item.paper_id}, chunk_index={item.chunk_index}, source={item.source})")


if __name__ == "__main__":
    main()

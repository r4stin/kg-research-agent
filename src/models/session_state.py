# src/models/session_state.py

from typing import List
from pydantic import BaseModel


class TurnMemory(BaseModel):
    """One Q&A turn in the conversation."""
    question: str
    answer: str  # full answer text (can be summarized later if needed)


class SessionState(BaseModel):
    """Session-level memory for one user/session."""
    turns: List[TurnMemory] = []

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(TurnMemory(question=question, answer=answer))

    def build_history_context(self, max_turns: int = 3) -> str:
        """
        Build a compact textual context from the last N turns.
        You can later add summarization instead of raw answers.
        """
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        blocks = []
        for i, t in enumerate(recent, start=1):
            # You can truncate long answers if needed
            blocks.append(
                f"[Turn {i}]\n"
                f"Q: {t.question}\n"
                f"A: {t.answer}\n"
            )
        return "\n".join(blocks)


# src/embeddings.py
from typing import List
import google.generativeai as genai
from chromadb.utils.embedding_functions import EmbeddingFunction

from src.config import GOOGLE_API_KEY  # note: src.config import

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in the environment.")

genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"  # you can tweak this later


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Chroma-compatible embedding function that uses Gemini embeddings.

    Why we inherit from EmbeddingFunction:
    - Chroma can introspect this object (e.g. name(), etc.)
    - It makes future changes safer as Chroma evolves.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Chroma calls this to embed lists of strings
        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[List[float]] = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
            )
            embeddings.append(result["embedding"])
        return embeddings

    def name(self) -> str:
        """
        Name identifier for this embedding function.

        Chroma uses this to check whether the same collection is being
        loaded with a different embedding configuration.
        """
        return f"gemini-{self.model}"

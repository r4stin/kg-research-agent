# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env once, at import time
load_dotenv()

# Base paths (repo root is one directory up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ==== Google / Gemini ====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-pro-exp")

# ==== Vector DB (Chroma) ====
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "data" / "chroma_db"))

# ==== Data paths ====
PDF_STORAGE = os.getenv("PDF_STORAGE", str(BASE_DIR / "data" / "papers"))
CHUNK_STORAGE = os.getenv("CHUNK_STORAGE", str(BASE_DIR / "data" / "chunks"))

# ==== Neo4j (not used yet, but ready) ====
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ==== Logging ====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))

Path(PDF_STORAGE).mkdir(parents=True, exist_ok=True)
Path(CHUNK_STORAGE).mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

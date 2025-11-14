# 📚 **KG-Research-Agent**
### *Evidence-Aware RAG + Knowledge Graph System Built with Gemini, ADK, ChromaDB & Neo4j*

<div align="center">

**🔥 A research-grade AI agent that extracts claims + evidence from scientific papers, stores them in a knowledge graph, and answers questions with fully grounded citations.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-blue.svg)]()
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple.svg)]()
[![Gemini](https://img.shields.io/badge/Gemini-LLM-orange.svg)]()

</div>

---

# 🚀 **Overview**

**KG-Research-Agent** is a modular, evidence-grounded research assistant designed to:

- Ingest scientific PDFs  
- Chunk + embed them using Gemini  
- Retrieve relevant evidence (RAG)  
- Extract claims + sentences with structured LLM outputs  
- Write them into a **Neo4j Knowledge Graph**  
- Answer questions with **verifiable citations**  
- Support **deduplication**, metadata tracking, and test coverage  


---

# 🧠 **High-Level Architecture**

```
                         ┌────────────────────────┐
                         │   User Research Query  │
                         └──────────────┬─────────┘
                                        │
                                        ▼
                            ┌────────────────────┐
                            │   Retriever Tool   │
                            │ (ChromaDB + Gemini)│
                            └────────────────────┘
                                        │ chunks
                                        ▼
                            ┌────────────────────┐
                            │ Evidence Extractor │
                            │ (LLM structured)   │
                            └────────────────────┘
                             │ claims + sentences
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Neo4j Knowledge Graph                    │
│──────────────────────────────────────────────────────────────────│
│ Nodes: Claim, Evidence, Paper, Chunk, Question                   │
│ Relationships: SUPPORTS, FROM_CHUNK, ANSWERS, IN_PAPER          │
└──────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                            ┌────────────────────┐
                            │  Answer Composer   │
                            │ with citations     │
                            └────────────────────┘
                                        │
                                        ▼
                               **Final Answer**
```

---

# ✨ **Current Features**

### ✔️ PDF → Chunking → Vector Store  
### ✔️ RAG Retrieval via ADK Tool  
### ✔️ LLM Structured Evidence Extraction  
### ✔️ Neo4j Knowledge Graph Writer  
### ✔️ KG Query Engine  
### ✔️ Deduplication (per chunk + similarity)  
### ✔️ Evidence → Answer Generation  
### ✔️ Full pytest suite  

---

# 🏁 **Getting Started**

## 1️⃣ Clone the Repo
```
git clone https://github.com/yourusername/kg-research-agent.git
cd kg-research-agent
```

## 2️⃣ Create Conda Environment
```
conda create -n kg-research-agent python=3.10
conda activate kg-research-agent
```

## 3️⃣ Install Requirements
```
pip install -r requirements.txt
```

## 4️⃣ Environment Variables (`.env`)

```
GOOGLE_API_KEY="your-key"
CHROMA_DB_PATH="data/chroma"
PDF_STORAGE="data/papers"

NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="yourpassword"
```

---

# 🧪 **Running the System**

### PDF Ingestion
```
python -m src.tools.pdf_ingest
```

### Evidence Extraction
```
python -m src.run_evidence_extraction
```

### KG Query
```
python -m src.pipelines.run_kg_query
```

### Full Answering Pipeline
```
python -m src.pipelines.run_evidence_and_answer
```

---

# 🧪 Tests

Run all tests:

```
pytest
```

Specifically dedup tests:

```
pytest tests/test_dedup_evidence_strict.py
```

---

# 🗺️ **Roadmap**

## 🟥 Agent Quality
- Add observability (logs, traces)
- Add metrics
- Add LLM-as-a-Judge evaluation
- Add scorecards for evidence quality

## 🟦 Multi-Agent System
- Planner agent
- Retriever agent
- Evidence agent
- Answer agent
- Message routing
- Modular ADK node design

## 🟩 Productionization
- Add A2A Protocol (Agent-to-Agent messaging)
- Deploy to Vertex AI Agent Engine
- Build HTTP endpoints
- Add scalable logging + monitoring

---

# 📜 License

MIT License.  
You may use, modify, and distribute this project freely.

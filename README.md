
# 📚 **KG-Research-Agent**
### *Multi-Agent, Evidence-Grounded Research System with Gemini, ADK, ChromaDB & Neo4j*

<div align="center">

**🔥 A research-grade AI agent that extracts claims + evidence from scientific papers, stores them in a knowledge graph, retrieves context, and answers questions using multi-agent reasoning with session memory.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-blue.svg)]()
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple.svg)]()
[![Gemini](https://img.shields.io/badge/Gemini-LLM-orange.svg)]()

</div>

---

# 🚀 **Overview**

**KG-Research-Agent** is an AI-powered research assistant that:

- Ingests scientific PDFs  
- Embeds + stores them in ChromaDB  
- Retrieves relevant text chunks (RAG)  
- Extracts **structured claims & evidence** from papers  
- Stores them in a **Neo4j Knowledge Graph**  
- Answers questions using **citations grounded in source text**  
- Uses a **multi-agent pipeline** (Planner → Retriever → Evidence → Answer)  
- Supports **multi-turn conversations with session memory**

A full walkthrough of the multi-agent research system is available on YouTube:

👉 **[Watch the Demo Video](https://youtu.be/vaq0-AMOudo)**

---

# 🧠 **Updated Architecture (Multi-Agent + Memory)**

```
┌──────── User ────────┐
          │
          ▼
┌───────────────┐
│ Planner Agent │  ← uses chat history + memory
└───────────────┘
     │ plans tasks
     ▼
┌────────────────────────┐
│ Retriever Agent        │ → ChromaDB (vector search)
└────────────────────────┘
     │ chunks
     ▼
┌────────────────────────┐
│ Evidence Agent         │ → extracts claims + sentences
└────────────────────────┘
     │ structured JSON
     ▼
┌────────────────────────┐
│ Answer Agent           │ → composes human-readable answer
└────────────────────────┘
     │
     ▼
 **Final Answer + Citations**

📦 Persistent Storage:
- Neo4j → long-term knowledge graph
- ChromaDB → vector retrieval
- SessionState → short-term conversation memory
```

---

# ✨ **Current Features**

### ✔️ PDF → Chunking → Vector Storage  
### ✔️ RAG Retrieval (Chroma + Gemini)  
### ✔️ Multi-Agent System (Planner → Retriever → Evidence → Answer)  
### ✔️ Structured JSON Evidence Extraction  
### ✔️ Neo4j Knowledge Graph Storage  
### ✔️ Session Memory (short-term conversational context)  
### ✔️ Deduplication (per chunk + semantic similarity)  
### ✔️ Multi-turn conversational research workflow  

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

# 🔧 **New: Multi-Agent Runner**

Run full pipeline with memory:

```
python -m src.pipelines.run_multi_agent_pipeline
```

Example:

```
You: What is a major challenge in scholarly information retrieval?
You: Summarize in one sentence.
```

The agent maintains context across turns.

---

# 🗺️ **Roadmap**

## 🟥 Agent Quality (Next Milestone)
- ADK logs + traces
- Metrics for agent performance
- LLM-as-a-Judge evaluation

## 🟦 Multi-Agent Enhancements
- Add **KG Agent** (read/write Neo4j in pipeline)
- Add planner task types: `kg_query`, `kg_write`
- Context compaction + memory optimization

## 🟩 Productionization
- A2A protocol (agent-to-agent messaging)
- Deployment to **Vertex AI Agent Engine**
- API endpoints + orchestration layer

---

# 📜 License

MIT License.  
You may use, modify, and distribute this project freely.

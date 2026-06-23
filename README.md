---
title: Healthcare Agentic RAG
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

Good base. Replace the entire file with this proper README:

```markdown
# Healthcare Agentic RAG

![CI](https://github.com/esther-jk7/healthcare-agentic-rag/actions/workflows/ci.yml/badge.svg)

A multi-agent clinical question-answering system that retrieves from PubMed literature, validates source quality, and generates cited answers — built with LangGraph, ChromaDB, and FastAPI.

## Architecture

```
User Question
      ↓
┌─────────────────┐
│  Retriever Agent │  → Queries ChromaDB vector store (2,918 PubMed chunks)
└─────────────────┘
      ↓
┌─────────────────┐
│  Validator Agent │  → Checks if retrieved chunks contain sufficient evidence
└─────────────────┘
      ↓ sufficient          ↓ insufficient (retry, max 2x)
┌─────────────────┐
│ Synthesizer Agent│  → Generates cited answer using Llama 3.3 70B via Groq
└─────────────────┘
      ↓
  JSON Response
  (answer + sources + low_confidence flag)
```

## Tech Stack

- **Orchestration:** LangGraph
- **Vector Store:** ChromaDB with `all-MiniLM-L6-v2` embeddings
- **LLM:** Llama 3.3 70B via Groq
- **API:** FastAPI with Pydantic validation
- **Observability:** LangSmith
- **Dataset:** PubMedQA (1,000 labeled samples, ~2,918 chunks)
- **CI/CD:** GitHub Actions (ruff lint + pytest)

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for documented findings including:
- Validator false-positive behavior and root cause
- Low confidence detection as an alternative to retry loops
- Embedding model selection rationale

## Quickstart

```bash
git clone https://github.com/esther-jk7/healthcare-agentic-rag
cd healthcare-agentic-rag
conda create -n mlsprint python=3.11
conda activate mlsprint
pip install -r requirements.txt
```

Add your API keys to `.env`:
```
GROQ_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=healthcare-agentic-rag
```

Build the vector store (one-time):
```bash
python -m src.embedder
```

Run the API:
```bash
uvicorn src.main:app --reload
```

Visit `http://localhost:8000/docs` to test.

## API

### POST /query
```json
{
  "question": "Do mitochondria play a role in programmed cell death?",
  "n_results": 5
}
```

Response:
```json
{
  "query": "Do mitochondria play a role in programmed cell death?",
  "answer": "Yes, mitochondria play a role... [PubMed 21645374]",
  "sources": [
    {"pubid": "21645374", "label": "BACKGROUND"},
    {"pubid": "21645374", "label": "RESULTS"}
  ],
  "low_confidence": false,
  "latency_ms": 1840.5
}
```

## Evaluation

- **Recall@5:** 1.00 on 10 samples (retrieval)
- **Low confidence detection:** correctly flags out-of-distribution queries
- **Latency:** ~1.2–2.5s per query

## Project Structure

```
src/
  agents.py      # LangGraph multi-agent graph
  embedder.py    # ChromaDB vector store builder
  rag.py         # Naive RAG pipeline
  main.py        # FastAPI application
  data_loader.py # PubMedQA data loader
tests/
  test_api.py    # Integration tests (6 passing)
notebooks/
  eda.ipynb      # Dataset analysis
  eval.py        # Recall@5 evaluation script
```
```

Save with **Cmd+S**. Has the CI run finished yet?
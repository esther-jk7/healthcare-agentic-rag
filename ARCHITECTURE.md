# Architecture Rationale

## Why Three Agents?

A single LLM call with retrieval (naive RAG) has one critical failure mode:
it generates an answer regardless of whether the retrieved chunks actually
contain relevant evidence. In clinical settings, a confident wrong answer
is worse than no answer.

Three agents enforce separation of concerns:
- **Retriever** — fast, deterministic, no LLM cost
- **Validator** — cheap binary decision (sufficient/insufficient)
- **Synthesizer** — expensive generation, only runs on validated evidence

This reduces unnecessary LLM calls and catches low-quality retrievals
before they reach generation.

## Why LangGraph over CrewAI or AutoGen?

LangGraph models the agent workflow as an explicit state machine with
typed state, conditional edges, and cycle support. This means:

- The retry loop (Validator → Retriever → Validator) is a first-class
  graph construct, not a hack
- Every state transition is inspectable via LangSmith traces
- Adding a new node (e.g. a Reranker) requires one `add_node` call
- The graph is deterministic — same input produces same execution path

CrewAI and AutoGen use higher-level abstractions that hide the execution
graph, making debugging harder and observability weaker.

## Why sentence-transformers over OpenAI embeddings?

`all-MiniLM-L6-v2` runs locally with zero API cost and zero latency
overhead from network calls. For a 1,000-sample dataset (~3,000 chunks),
the quality difference vs `text-embedding-3-small` is marginal for
medical Q&A retrieval (our Recall@5 = 1.00 on test samples).

In production, switching to a higher-quality embedding model is a
one-line change in `src/embedder.py`.

## Failure Modes and Mitigations

### 1. Validator false positives
**Problem:** Validator says "sufficient" but chunks don't directly answer
the question. Observed with mammography screening query — retrieved
chunks discussed mammography interventions for a specific population,
not effectiveness evidence.

**Root cause:** Semantic similarity retrieves topically adjacent chunks,
not question-specific ones. Data coverage gap in PubMedQA (1,000 samples
is insufficient for broad effectiveness questions).

**Mitigation:** Low confidence detection in Synthesizer. If the answer
contains phrases like "no direct evidence" or "insufficient evidence",
`low_confidence=true` is returned to the caller. More honest than
forcing retry loops against the same corpus.

### 2. Infinite retry loops
**Problem:** If Validator always returns "insufficient", the graph loops
forever.

**Mitigation:** `retry_count` in AgentState caps retries at 2.
After 2 retries, Synthesizer runs regardless and returns
`low_confidence=true` if evidence is weak.

### 3. Context window overflow
**Problem:** Long chunks + many results could exceed the LLM context window.

**Mitigation:** 20-word minimum filter removes noise chunks. n_results
capped at 10 via Pydantic validation. Current average chunk = 60 words,
so 10 chunks ≈ 600 words — well within Llama 3.3 70B's 128K context.

## What I Would Do Differently in Production

1. Replace `all-MiniLM-L6-v2` with a medical-domain embedding model
   (e.g. BiomedBERT embeddings) for better clinical text retrieval
2. Add a reranker node between Retriever and Validator
   (cross-encoder reranking improves precision@k significantly)
3. Use persistent vector store (Pinecone or Weaviate) instead of
   local ChromaDB for horizontal scaling
4. Add HIPAA-compliant audit logging for every query in clinical deployment
5. Replace Groq with Azure OpenAI for enterprise SLA guarantees
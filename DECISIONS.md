# Engineering Decisions & Findings

## Validator Logic (Jun 9, 2026)
**Finding:** Stricter Validator prompt alone does not fix false-positive 
"sufficient" verdicts. The LLM sees topically related keywords 
(e.g. "mammography") and calls chunks sufficient even when they 
don't answer the specific question.

**Root cause:** `all-MiniLM-L6-v2` retrieves semantically adjacent 
chunks, not question-specific ones. Data coverage gap in PubMedQA 
(1,000 samples insufficient for broad "does X work" questions).

**Decision:** Added low_confidence detection in Synthesizer instead 
of forcing retry loops. More honest than retrying with same query 
against same corpus.

**Recall@5 baseline:** 1.00 on 10 samples (retrieval works; 
answering is harder).

## Confidence Scoring (Jun 9, 2026)
**Finding:** 3/5 test queries flagged low_confidence=True. 
PubMedQA skews toward specific disease/intervention studies, 
not broad effectiveness questions.

**Decision:** Expose low_confidence flag in QueryResponse so 
callers can handle uncertainty appropriately.

## HuggingFace Spaces Deployment (Jun 23, 2026)

**Problem 1:** ChromaDB vector store doesn't persist between container restarts on HF Spaces.
**Solution:** `startup.sh` checks for `chroma_db/` on every startup. If missing, downloads 
PubMedQA from HuggingFace datasets and rebuilds the vector store before starting the API.
This adds ~3 minutes to cold start but requires zero external storage.

**Problem 2:** `data/pubmedqa_train.json` is in `.gitignore` so it never gets pushed.
**Solution:** Startup script downloads directly from `qiaojin/PubMedQA` via HuggingFace 
datasets library instead of reading a local file.

**Problem 3:** API secrets (GROQ_API_KEY, LANGCHAIN_API_KEY) can't be baked into the image.
**Solution:** HuggingFace Spaces "Variables and secrets" — injected as environment variables 
at runtime. Never in code, never in the image.

**Live URL:** https://estherrjk-healthcare-agentic-rag.hf.space/docs
**Cold start time:** ~3-4 minutes (dataset download + embedding)
**Warm response time:** ~1.2-2.5s per query
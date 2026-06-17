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
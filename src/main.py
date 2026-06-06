from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import naive_rag

app = FastAPI(title="Healthcare Agentic RAG")

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5

class Source(BaseModel):
    pubid: str
    label: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = naive_rag(request.question, request.n_results)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]]
    )
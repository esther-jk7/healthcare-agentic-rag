from fastapi import FastAPI
from pydantic import BaseModel
from src.agents import build_graph

app = FastAPI(title="Healthcare Agentic RAG")

# Build the agent graph once at startup
agent_graph = build_graph()

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
    low_confidence: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = agent_graph.invoke({
        "question": request.question,
        "chunks": [],
        "validation_status": "pending",
        "answer": "",
        "retry_count": 0,
        "low_confidence": False
    })
    return QueryResponse(
        query=result["question"],
        answer=result["answer"],
        sources=[
            Source(pubid=c["pubid"], label=c["label"])
            for c in result["chunks"]
        ],
        low_confidence=result.get("low_confidence", False)
    )
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.agents import build_graph

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Agentic RAG",
    description="Multi-agent clinical Q&A system with citation validation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Build agent graph once at startup
agent_graph = build_graph()
logger.info("Agent graph initialized successfully")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Medical question to answer"
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of chunks to retrieve"
    )

class Source(BaseModel):
    pubid: str
    label: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    low_confidence: bool
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()
    logger.info(f"Query received: {request.question[:60]}...")

    try:
        result = agent_graph.invoke({
            "question": request.question,
            "chunks": [],
            "validation_status": "pending",
            "answer": "",
            "retry_count": 0,
            "low_confidence": False
        })
    except Exception as e:
        logger.error(f"Agent graph failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    if not result.get("answer"):
        raise HTTPException(status_code=422, detail="Agent returned empty answer")

    latency_ms = (time.time() - start) * 1000
    logger.info(f"Query completed in {latency_ms:.0f}ms | low_confidence={result.get('low_confidence')}")

    return QueryResponse(
        query=result["question"],
        answer=result["answer"],
        sources=[
            Source(pubid=c["pubid"], label=c["label"])
            for c in result["chunks"]
        ],
        low_confidence=result.get("low_confidence", False),
        latency_ms=round(latency_ms, 2)
    )
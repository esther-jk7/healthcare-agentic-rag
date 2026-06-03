from fastapi import FastAPI

app = FastAPI(title="Healthcare Agentic RAG")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Healthcare Agentic RAG API is running"}

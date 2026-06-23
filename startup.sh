#!/bin/bash
echo "Checking vector store..."
if [ ! -d "chroma_db" ]; then
    echo "Downloading PubMedQA and building vector store..."
    python -m src.embedder
fi
echo "Starting API..."
uvicorn src.main:app --host 0.0.0.0 --port 7860

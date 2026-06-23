#!/bin/bash
set -e
echo "Checking data..."
if [ ! -f "data/pubmedqa_train.json" ]; then
    echo "Downloading PubMedQA dataset..."
    mkdir -p data
    python -c "
from datasets import load_dataset
import json
dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')
with open('data/pubmedqa_train.json', 'w') as f:
    json.dump(list(dataset['train']), f)
print('Dataset downloaded.')
"
fi
echo "Checking vector store..."
if [ ! -d "chroma_db" ]; then
    echo "Building vector store..."
    python -m src.embedder
fi
echo "Starting API..."
uvicorn src.main:app --host 0.0.0.0 --port 7860

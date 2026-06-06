import os
from groq import Groq
from dotenv import load_dotenv
from src.embedder import get_embedding, get_collection

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """Retrieve top-n relevant chunks from ChromaDB."""
    collection = get_collection()
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "pubid": meta["pubid"],
            "label": meta["label"]
        })
    return chunks


def generate(query: str, chunks: list[dict]) -> str:
    """Generate an answer using retrieved chunks as context."""
    context = "\n\n".join([
        f"[Source: PubMed {c['pubid']} - {c['label']}]\n{c['text']}"
        for c in chunks
    ])
    
    prompt = f"""You are a medical research assistant. Answer the question based only on the provided context. 
Always cite the PubMed ID of the source you used.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return response.choices[0].message.content


def naive_rag(query: str, n_results: int = 5) -> dict:
    """Full naive RAG pipeline: retrieve + generate."""
    chunks = retrieve(query, n_results)
    answer = generate(query, chunks)
    
    return {
        "query": query,
        "answer": answer,
        "sources": [{"pubid": c["pubid"], "label": c["label"]} for c in chunks]
    }


if __name__ == "__main__":
    query = "Do mitochondria play a role in programmed cell death?"
    result = naive_rag(query)
    
    print(f"Query: {result['query']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used:")
    for s in result["sources"]:
        print(f"  - PubMed {s['pubid']} ({s['label']})")
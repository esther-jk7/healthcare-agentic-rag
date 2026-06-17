import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.data_loader import load_pubmedqa

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()

def build_vector_store(
    data_path: str = "data/pubmedqa_train.json",
    collection_name: str = "pubmedqa",
    min_chunk_words: int = 20
) -> chromadb.Collection:

    chroma_client = chromadb.PersistentClient(path="chroma_db")

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(collection_name)
    samples = load_pubmedqa(data_path)

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    print("Embedding chunks...")
    skipped = 0

    for sample in samples:
        for i, chunk in enumerate(sample["chunks"]):
            text = chunk["text"]

            if len(text.split()) < min_chunk_words:
                skipped += 1
                continue

            embedding = get_embedding(text)
            chunk_id = f"{sample['pubid']}_{i}"

            documents.append(text)
            embeddings.append(embedding)
            metadatas.append({
                "pubid": str(sample["pubid"]),
                "label": chunk["label"],
                "question": sample["question"],
                "final_decision": sample["final_decision"]
            })
            ids.append(chunk_id)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Done. Stored {len(documents)} chunks. Skipped {skipped} short chunks.")
    return collection


def get_collection(collection_name: str = "pubmedqa") -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    return chroma_client.get_collection(collection_name)


if __name__ == "__main__":
    collection = build_vector_store()

    test_query = "Do mitochondria play a role in programmed cell death?"
    query_embedding = get_embedding(test_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\nTest query: {test_query}")
    print("\nTop 3 results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n[{i+1}] PubID: {meta['pubid']} | Label: {meta['label']}")
        print(f"    {doc[:150]}...")
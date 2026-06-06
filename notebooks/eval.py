import sys
sys.path.append('.')
import json
from src.rag import retrieve
from src.embedder import get_embedding, get_collection

def recall_at_k(retrieved_pubids: list[str], relevant_pubid: str, k: int = 5) -> float:
    """
    Recall@k: did we retrieve the correct paper in the top-k results?
    Returns 1.0 if yes, 0.0 if no.
    """
    return 1.0 if relevant_pubid in retrieved_pubids[:k] else 0.0

def run_eval(n_samples: int = 10, k: int = 5):
    with open("data/pubmedqa_train.json") as f:
        samples = json.load(f)

    test_samples = samples[:n_samples]
    
    scores = []
    results = []

    print(f"Running evaluation on {n_samples} samples (Recall@{k})\n")
    print("-" * 60)

    for i, sample in enumerate(test_samples):
        question = sample["question"]
        correct_pubid = str(sample["pubid"])
        
        chunks = retrieve(question, n_results=k)
        retrieved_pubids = [c["pubid"] for c in chunks]
        
        score = recall_at_k(retrieved_pubids, correct_pubid, k)
        scores.append(score)
        
        status = "✓" if score == 1.0 else "✗"
        print(f"[{status}] Sample {i+1}: {question[:60]}...")
        print(f"    Correct PubID: {correct_pubid}")
        print(f"    Retrieved: {retrieved_pubids}")
        print()
        
        results.append({
            "question": question,
            "correct_pubid": correct_pubid,
            "retrieved_pubids": retrieved_pubids,
            "recall_at_k": score
        })

    avg_recall = sum(scores) / len(scores)
    print("-" * 60)
    print(f"Recall@{k} on {n_samples} samples: {avg_recall:.2f} ({sum(scores)}/{n_samples} correct)")
    
    with open("notebooks/eval_results.json", "w") as f:
        json.dump({
            "recall_at_k": avg_recall,
            "k": k,
            "n_samples": n_samples,
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to notebooks/eval_results.json")
    return avg_recall

if __name__ == "__main__":
    run_eval(n_samples=10, k=5)
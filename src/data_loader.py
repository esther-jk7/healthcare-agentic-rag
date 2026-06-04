import json
from pathlib import Path

def load_pubmedqa(path: str = "data/pubmedqa_train.json") -> list[dict]:
    """
    Loads PubMedQA dataset from local JSON file.
    Returns a list of samples, each with question, contexts, answer, and decision.
    """
    with open(Path(path)) as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        # Extract the text chunks from context
        contexts = item["context"]["contexts"]
        labels = item["context"]["labels"]

        # Pair each chunk with its section label
        chunks = [
            {"text": ctx, "label": lbl, "pubid": item["pubid"]}
            for ctx, lbl in zip(contexts, labels)
        ]

        samples.append({
            "pubid": item["pubid"],
            "question": item["question"],
            "chunks": chunks,
            "long_answer": item["long_answer"],
            "final_decision": item["final_decision"]
        })

    return samples


if __name__ == "__main__":
    samples = load_pubmedqa()
    print(f"Loaded {len(samples)} samples")
    print(f"\nFirst question: {samples[0]['question']}")
    print(f"Number of chunks: {len(samples[0]['chunks'])}")
    print(f"First chunk label: {samples[0]['chunks'][0]['label']}")
    print(f"Expected answer: {samples[0]['final_decision']}")

from datasets import load_dataset
import json
import os

print("Downloading PubMedQA dataset...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)

os.makedirs("data", exist_ok=True)

with open("data/pubmedqa_train.json", "w") as f:
    json.dump(list(dataset["train"]), f, indent=2)

print(f"Done! {len(dataset['train'])} samples saved to data/pubmedqa_train.json")

"""Builds data/prompts.jsonl: 1000 records sampled from databricks-dolly-15k (CC BY-SA 3.0).

Usage: curl -L -o dolly.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
       python data/prepare_dataset.py dolly.jsonl
"""
import json, random, sys
from pathlib import Path

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
random.Random(42).shuffle(rows)
out = Path(__file__).parent / "prompts.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for i, r in enumerate(rows[:1000]):
        prompt = r["instruction"] + (f"\n\nContext:\n{r['context']}" if r["context"] else "")
        f.write(json.dumps({"id": i, "category": r["category"], "prompt": prompt, "response": r["response"]}) + "\n")
print(f"wrote {out}")

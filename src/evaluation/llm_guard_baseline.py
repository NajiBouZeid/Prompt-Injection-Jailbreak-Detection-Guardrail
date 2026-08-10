"""Runs LLM Guard's PromptInjection scanner over the same 1439-row Gemini-judge
sample used elsewhere in this project, for a fair 3-way comparison against
DeBERTa and Gemini. Must be run with venv-llmguard's python, not the main
project venv - llm-guard's dependency tree conflicts with the pinned
transformers/huggingface-hub versions the DeBERTa pipeline needs.

Writes one JSON line per row to an output file (resumable, same pattern as
gemini_judge.py) rather than holding everything in memory, since the scanner
loads its own ~184M-param model per process.
"""
import json
import os

import pandas as pd
from llm_guard.input_scanners import PromptInjection

SAMPLE_PATH = "data/processed/gemini_judge_sample.parquet"


def load_done_ids(results_path: str) -> set:
    if not os.path.exists(results_path):
        return set()
    with open(results_path) as f:
        return {json.loads(line)["row_id"] for line in f if line.strip()}


def run(sample_path: str = SAMPLE_PATH, output_path: str = "data/processed/llm_guard_results.jsonl"):
    sample = pd.read_parquet(sample_path)
    done_ids = load_done_ids(output_path)
    todo = sample[~sample.index.isin(done_ids)]

    print(f"{len(done_ids)} rows already scored, {len(todo)} remaining")
    if len(todo) == 0:
        return

    scanner = PromptInjection()

    with open(output_path, "a") as f:
        for i, (row_id, row) in enumerate(todo.iterrows(), start=1):
            _, is_valid, risk_score = scanner.scan(row["text"])
            result = {
                "row_id": int(row_id),
                "predicted_label": 0 if is_valid else 1,
                "risk_score": float(risk_score),
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] scored so far: {len(done_ids) + i}/{len(sample)}")

    print(f"done: {len(sample)}/{len(sample)} rows scored")

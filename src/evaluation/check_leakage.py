import re
import time

import numpy as np
import pandas as pd
import torch
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

from src.evaluation.evaluate_classifier import compute_binary_metrics, load_model, predict_dataframe

TEST_SAMPLE_SIZE = 2000
RANDOM_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SHINGLE_SIZE = 5  # words per shingle
NUM_PERM = 128
LSH_THRESHOLD = 0.5  # Jaccard cutoff for LSH candidate generation (exact Jaccard computed after)


def load_necent_split(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["source_dataset"] == "necent"].reset_index(drop=True)


def shingle(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def make_minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for sh in shingle(text):
        mh.update(sh.encode("utf8"))
    return mh


def embedding_max_similarity(
    train_texts: list[str], test_texts: list[str], batch_size: int = 256, chunk: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """For each test text, the max cosine similarity to any train text (+ its index)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    train_emb = model.encode(
        train_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    test_emb = model.encode(
        test_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    max_sim = np.full(len(test_texts), -1.0, dtype=np.float32)
    best_idx = np.full(len(test_texts), -1, dtype=np.int64)
    for start in range(0, len(train_emb), chunk):
        block = train_emb[start : start + chunk]
        sims = test_emb @ block.T
        block_max = sims.max(axis=1)
        block_argmax = sims.argmax(axis=1) + start
        better = block_max > max_sim
        max_sim[better] = block_max[better]
        best_idx[better] = block_argmax[better]

    return max_sim, best_idx


def minhash_max_jaccard(
    train_texts: list[str], test_texts: list[str], log_every: int = 20000
) -> tuple[np.ndarray, list[int]]:
    """For each test text, the best estimated Jaccard similarity found via LSH candidate
    lookup against the full train set."""
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    train_minhashes = []
    start = time.time()
    for i, text in enumerate(train_texts):
        mh = make_minhash(text)
        train_minhashes.append(mh)
        lsh.insert(str(i), mh)
        if (i + 1) % log_every == 0:
            print(f"[minhash index] {i + 1}/{len(train_texts)} ({time.time() - start:.0f}s)", flush=True)

    best_jaccard = np.zeros(len(test_texts), dtype=np.float32)
    best_train_idx = [-1] * len(test_texts)
    for i, text in enumerate(test_texts):
        mh = make_minhash(text)
        candidates = lsh.query(mh)
        if not candidates:
            continue
        best = max(candidates, key=lambda c: mh.jaccard(train_minhashes[int(c)]))
        best_jaccard[i] = mh.jaccard(train_minhashes[int(best)])
        best_train_idx[i] = int(best)

    return best_jaccard, best_train_idx


def run(
    train_path: str = "data/processed/train.parquet",
    test_path: str = "data/processed/test.parquet",
    test_sample_size: int = TEST_SAMPLE_SIZE,
    model_dir: str | None = None,
) -> dict:
    train_df = load_necent_split(train_path)
    test_df = load_necent_split(test_path)

    rng = np.random.default_rng(RANDOM_SEED)
    n = min(test_sample_size, len(test_df))
    sample_idx = rng.choice(len(test_df), size=n, replace=False)
    test_sample = test_df.iloc[sample_idx].reset_index(drop=True)

    print(f"Embedding pass: {len(train_df)} train vs {len(test_sample)} test...", flush=True)
    emb_sim, emb_idx = embedding_max_similarity(train_df["text"].tolist(), test_sample["text"].tolist())

    print(f"MinHash pass: {len(train_df)} train vs {len(test_sample)} test...", flush=True)
    mh_jaccard, mh_idx = minhash_max_jaccard(train_df["text"].tolist(), test_sample["text"].tolist())

    report = {
        "n_train": len(train_df),
        "n_test_sampled": len(test_sample),
        "embedding": {
            "pct_above_0.99": float((emb_sim >= 0.99).mean()),
            "pct_above_0.95": float((emb_sim >= 0.95).mean()),
            "pct_above_0.90": float((emb_sim >= 0.90).mean()),
            "pct_above_0.80": float((emb_sim >= 0.80).mean()),
            "mean_max_sim": float(emb_sim.mean()),
        },
        "minhash": {
            "pct_above_0.9": float((mh_jaccard >= 0.9).mean()),
            "pct_above_0.8": float((mh_jaccard >= 0.8).mean()),
            "pct_above_0.5": float((mh_jaccard >= 0.5).mean()),
            "pct_zero_candidates": float((mh_jaccard == 0).mean()),
        },
    }

    if model_dir is not None:
        print(f"Scoring DeBERTa on the same {len(test_sample)}-row sample...", flush=True)
        model, tokenizer, device = load_model(model_dir)
        predictions, probs = predict_dataframe(model, tokenizer, test_sample, device)
        labels = test_sample["label"].to_numpy()

        leakage_adjusted = {}
        masks = {
            "leaked_embedding_ge_0.99": emb_sim >= 0.99,
            "leaked_embedding_ge_0.95": emb_sim >= 0.95,
            "clean_embedding_lt_0.95": emb_sim < 0.95,
        }
        for name, mask in masks.items():
            if mask.sum() == 0:
                continue
            leakage_adjusted[name] = compute_binary_metrics(
                labels[mask], predictions[mask], probs[mask]
            )
        report_leakage_adjusted = leakage_adjusted
    else:
        report_leakage_adjusted = None

    top_n = 15
    top_emb_idx = np.argsort(-emb_sim)[:top_n]
    report["top_embedding_matches"] = [
        {
            "test_text": test_sample.iloc[int(i)]["text"][:300],
            "train_text": train_df.iloc[int(emb_idx[i])]["text"][:300],
            "embedding_sim": float(emb_sim[i]),
            "minhash_jaccard": float(mh_jaccard[i]),
        }
        for i in top_emb_idx
    ]
    report["leakage_adjusted_metrics"] = report_leakage_adjusted

    return report

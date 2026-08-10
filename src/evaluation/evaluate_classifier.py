import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

MAX_LENGTH = 512


def load_test_split(test_split: str = "test", max_examples: int | None = None) -> pd.DataFrame:
    """Load a processed split with all metadata columns kept (unlike training's
    load_split, which drops everything but text/label) - source_dataset is needed
    here for the per-source stratified breakdown below."""
    df = pd.read_parquet(f"data/processed/{test_split}.parquet")
    if max_examples is not None:
        df = df.iloc[:max_examples]
    return df


def compute_binary_metrics(
    labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray | None = None
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    accuracy = (predictions == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "confusion_matrix": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        },
        "n": int(len(labels)),
    }
    # AUC metrics need both classes present and are undefined for tiny/degenerate groups.
    if probs is not None and len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    return metrics


def measure_latency(
    model, tokenizer, texts: list[str], device: torch.device, n_samples: int = 200
) -> dict:
    """Single-example (batch size 1) inference latency - deliberately separate
    from the batched throughput of the main predict loop below, since
    batch-size-1 is what a request-at-a-time guardrail deployment would see."""
    model.eval()
    sample = texts[:n_samples]
    latencies_ms = []
    with torch.no_grad():
        for text in sample:
            inputs = tokenizer(
                text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            start = time.perf_counter()
            model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_samples": len(arr),
    }


def load_model(model_dir: str) -> tuple:
    """Loads a saved model/checkpoint + tokenizer onto the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer, device


def predict_dataframe(
    model, tokenizer, df: pd.DataFrame, device: torch.device, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over df's texts. Returns (hard predictions, positive-class
    probabilities) aligned with df's row order - shared by both the standard
    test-split eval and any other df of texts (e.g. the Gemini judge sample)."""
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text", "label"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(
        tokenized, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer)
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(all_logits)
    predictions = np.argmax(logits, axis=-1)

    # Softmax probability of the positive (attack) class, used for ROC/PR-AUC -
    # those need a ranking score, not just the hard 0/1 prediction argmax gives.
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    softmax = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    probs_positive = softmax[:, 1]

    return predictions, probs_positive


def evaluate(
    model_dir: str,
    test_split: str = "test",
    batch_size: int = 32,
    max_examples: int | None = None,
    latency_samples: int = 200,
) -> dict:
    """Full evaluation: overall metrics, metrics stratified by source_dataset,
    and single-example inference latency. Returns a plain JSON-serializable
    dict rather than printing, so callers can save it, diff it against a
    previous run's report, or feed it into a DeBERTa-vs-LLM-judge comparison."""
    model, tokenizer, device = load_model(model_dir)

    df = load_test_split(test_split, max_examples)
    predictions, probs_positive = predict_dataframe(model, tokenizer, df, device, batch_size)
    labels = df["label"].to_numpy()

    # Cross-entropy loss, comparable to the `loss` HF logs during training,
    # to sanity-check train vs. eval loss for overfitting.
    log_probs = np.log(np.clip(probs_positive * labels + (1 - probs_positive) * (1 - labels), 1e-12, 1.0))
    eval_loss = float(-log_probs.mean())

    report = {
        "eval_loss": eval_loss,
        "overall": compute_binary_metrics(labels, predictions, probs_positive),
    }

    df_eval = df.copy()
    df_eval["prediction"] = predictions
    df_eval["prob_positive"] = probs_positive
    for source, group in df_eval.groupby("source_dataset"):
        report[f"source__{source}"] = compute_binary_metrics(
            group["label"].to_numpy(),
            group["prediction"].to_numpy(),
            group["prob_positive"].to_numpy(),
        )

    report["latency"] = measure_latency(
        model, tokenizer, df["text"].tolist(), device, latency_samples
    )

    return report

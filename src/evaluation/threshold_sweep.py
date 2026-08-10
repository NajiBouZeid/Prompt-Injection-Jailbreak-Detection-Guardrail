import numpy as np

from src.evaluation.evaluate_classifier import (
    compute_binary_metrics,
    load_model,
    load_test_split,
    predict_dataframe,
)


def sweep(model_dir: str, split: str = "val", batch_size: int = 32) -> dict:
    """Sweeps a single global decision threshold (applied uniformly, not
    per-source - source_dataset isn't available at real inference time) over
    softmax probabilities, tuned on `split` (val by default, to avoid tuning
    on the same test set used for the final reported numbers). Returns
    overall + per-source metrics at each threshold plus probability-only
    stats needed to judge separability."""
    model, tokenizer, device = load_model(model_dir)
    df = load_test_split(split)
    _, probs = predict_dataframe(model, tokenizer, df, device, batch_size)

    df = df.copy()
    df["prob_positive"] = probs
    labels = df["label"].to_numpy()

    thresholds = np.round(
        np.concatenate([
            np.arange(0.05, 0.90, 0.05),
            np.arange(0.90, 0.99, 0.01),
            np.arange(0.99, 0.9999, 0.002),
        ]), 4
    )
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        entry = {"threshold": float(t), "overall": compute_binary_metrics(labels, preds, probs)}
        for source, group in df.groupby("source_dataset"):
            group_preds = (group["prob_positive"].to_numpy() >= t).astype(int)
            entry[f"source__{source}"] = compute_binary_metrics(
                group["label"].to_numpy(), group_preds, group["prob_positive"].to_numpy()
            )
        results.append(entry)

    return {"split": split, "n": len(df), "sweep": results}


def best_threshold_for_source(sweep_report: dict, source: str, metric: str = "f1") -> dict:
    """Picks the threshold maximizing `metric` on the given source's group."""
    key = f"source__{source}"
    return max(sweep_report["sweep"], key=lambda entry: entry[key][metric])

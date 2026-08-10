import json

import pandas as pd

from src.evaluation.evaluate_classifier import (
    compute_binary_metrics,
    load_model,
    predict_dataframe,
)

SAMPLE_PATH = "data/processed/gemini_judge_sample.parquet"
GEMINI_RESULTS_PATH = "data/processed/gemini_judge_results.jsonl"

LABEL_MAP = {"attack": 1, "benign": 0}


def load_gemini_predictions(results_path: str) -> pd.DataFrame:
    """Loads Gemini's per-row_id results, indexed to align 1:1 with
    gemini_judge_sample.parquet's row order."""
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame(rows).set_index("row_id").sort_index()


def compare(model_dir: str, batch_size: int = 32) -> dict:
    """Compares DeBERTa against Gemini on the same 1439-row sample. Rows where
    Gemini's safety filter refused to answer are excluded from Gemini's metrics
    (not scored as right or wrong) but reported separately as a refusal rate,
    since a refusal isn't a classification - see gemini_judge results for detail."""
    sample = pd.read_parquet(SAMPLE_PATH)
    gemini = load_gemini_predictions(GEMINI_RESULTS_PATH)
    if len(gemini) != len(sample):
        raise RuntimeError(
            f"expected {len(sample)} judged rows, found {len(gemini)} - "
            "is the Gemini judge run still in progress?"
        )

    model, tokenizer, device = load_model(model_dir)
    deberta_pred, deberta_prob = predict_dataframe(model, tokenizer, sample, device, batch_size)

    df = sample.copy()
    df["deberta_prediction"] = deberta_pred
    df["deberta_prob_positive"] = deberta_prob
    df["gemini_refused"] = gemini["predicted_label"].isna()
    df["gemini_prediction"] = gemini["predicted_label"].map(LABEL_MAP)

    report = {
        "n_total": len(df),
        "n_gemini_refused": int(df["gemini_refused"].sum()),
        "deberta_overall": compute_binary_metrics(
            df["label"].to_numpy(), df["deberta_prediction"].to_numpy(), df["deberta_prob_positive"].to_numpy()
        ),
        "gemini_overall": compute_binary_metrics(
            df.loc[~df["gemini_refused"], "label"].to_numpy(),
            df.loc[~df["gemini_refused"], "gemini_prediction"].to_numpy().astype(int),
        ),
    }

    for source, group in df.groupby("source_dataset"):
        answered = group[~group["gemini_refused"]]
        report[f"deberta_source__{source}"] = compute_binary_metrics(
            group["label"].to_numpy(), group["deberta_prediction"].to_numpy(), group["deberta_prob_positive"].to_numpy()
        )
        report[f"gemini_source__{source}"] = compute_binary_metrics(
            answered["label"].to_numpy(), answered["gemini_prediction"].to_numpy().astype(int)
        )
        report[f"refusals__{source}"] = {
            "n_refused": int(group["gemini_refused"].sum()),
            "n_total": int(len(group)),
        }

    answered = df[~df["gemini_refused"]]
    report["agreement_rate"] = float(
        (answered["deberta_prediction"] == answered["gemini_prediction"]).mean()
    )

    disagreements = answered[answered["deberta_prediction"] != answered["gemini_prediction"]]
    report["disagreements"] = [
        {
            "row_id": int(idx),
            "source_dataset": row["source_dataset"],
            "true_label": int(row["label"]),
            "deberta_prediction": int(row["deberta_prediction"]),
            "gemini_prediction": int(row["gemini_prediction"]),
            "text_preview": row["text"][:200],
        }
        for idx, row in disagreements.iterrows()
    ]

    return report

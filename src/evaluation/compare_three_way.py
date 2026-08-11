import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.evaluation.compare_judges import GEMINI_RESULTS_PATH, LABEL_MAP, load_gemini_predictions
from src.evaluation.evaluate_classifier import compute_binary_metrics, load_model, predict_dataframe

SAMPLE_PATH = "data/processed/gemini_judge_sample.parquet"
LLM_GUARD_RESULTS_PATH = "data/processed/llm_guard_results.jsonl"


def load_llm_guard_predictions(results_path: str) -> pd.DataFrame:
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame(rows).set_index("row_id").sort_index()


def compare(model_dir: str, batch_size: int = 32) -> dict:
    """Compares DeBERTa, Gemini, and LLM Guard's PromptInjection scanner on the
    same 1439-row sample used throughout this project. LLM Guard has no refusal
    concept (unlike Gemini) so all its rows are always scored."""
    sample = pd.read_parquet(SAMPLE_PATH)
    gemini = load_gemini_predictions(GEMINI_RESULTS_PATH)
    llm_guard = load_llm_guard_predictions(LLM_GUARD_RESULTS_PATH)
    for name, other in [("gemini", gemini), ("llm_guard", llm_guard)]:
        if len(other) != len(sample):
            raise RuntimeError(
                f"expected {len(sample)} rows, found {len(other)} for {name} - is that run still in progress?"
            )

    model, tokenizer, device = load_model(model_dir)
    deberta_pred, deberta_prob = predict_dataframe(model, tokenizer, sample, device, batch_size)

    df = sample.copy()
    df["deberta_prediction"] = deberta_pred
    df["deberta_prob_positive"] = deberta_prob
    df["gemini_refused"] = gemini["predicted_label"].isna()
    df["gemini_prediction"] = gemini["predicted_label"].map(LABEL_MAP)
    df["llm_guard_prediction"] = llm_guard["predicted_label"]
    df["llm_guard_risk_score"] = llm_guard["risk_score"]

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
        "llm_guard_overall": compute_binary_metrics(
            df["label"].to_numpy(), df["llm_guard_prediction"].to_numpy()
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
        report[f"llm_guard_source__{source}"] = compute_binary_metrics(
            group["label"].to_numpy(), group["llm_guard_prediction"].to_numpy()
        )

    # Pairwise Cohen's Kappa between all three judges - how much the judges
    # agree with EACH OTHER (not with ground truth), corrected for chance
    # agreement. Gemini's refused rows are excluded from any pair it's in,
    # same rule used for its own accuracy metrics above.
    answered = df[~df["gemini_refused"]]
    report["pairwise_agreement"] = {
        "deberta_vs_gemini": {
            "cohen_kappa": float(cohen_kappa_score(answered["deberta_prediction"], answered["gemini_prediction"])),
            "raw_agreement_rate": float((answered["deberta_prediction"] == answered["gemini_prediction"]).mean()),
            "n": int(len(answered)),
        },
        "deberta_vs_llm_guard": {
            "cohen_kappa": float(cohen_kappa_score(df["deberta_prediction"], df["llm_guard_prediction"])),
            "raw_agreement_rate": float((df["deberta_prediction"] == df["llm_guard_prediction"]).mean()),
            "n": int(len(df)),
        },
        "gemini_vs_llm_guard": {
            "cohen_kappa": float(cohen_kappa_score(answered["gemini_prediction"].astype(int), answered["llm_guard_prediction"])),
            "raw_agreement_rate": float((answered["gemini_prediction"].astype(int) == answered["llm_guard_prediction"]).mean()),
            "n": int(len(answered)),
        },
    }

    return report

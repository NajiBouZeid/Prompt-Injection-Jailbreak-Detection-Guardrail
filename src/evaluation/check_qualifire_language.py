import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from src.evaluation.evaluate_classifier import load_model, predict_dataframe

# langdetect's detection is non-deterministic across runs unless seeded.
DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def run(model_dir: str, batch_size: int = 32) -> dict:
    """Checks whether qualifire's false positives skew towards non-English text -
    a plausible explanation for DeBERTa's known qualifire precision problem, since
    the base model (deberta-v3-base) is English-only."""
    df = pd.read_parquet("data/processed/test.parquet")
    qualifire = df[df["source_dataset"] == "qualifire"].reset_index(drop=True)

    model, tokenizer, device = load_model(model_dir)
    predictions, probs_positive = predict_dataframe(model, tokenizer, qualifire, device, batch_size)

    qualifire = qualifire.copy()
    qualifire["prediction"] = predictions
    qualifire["prob_positive"] = probs_positive
    qualifire["language"] = qualifire["text"].apply(detect_language)

    qualifire["outcome"] = "tn"
    qualifire.loc[(qualifire["label"] == 1) & (qualifire["prediction"] == 1), "outcome"] = "tp"
    qualifire.loc[(qualifire["label"] == 0) & (qualifire["prediction"] == 1), "outcome"] = "fp"
    qualifire.loc[(qualifire["label"] == 1) & (qualifire["prediction"] == 0), "outcome"] = "fn"

    lang_counts = qualifire["language"].value_counts().to_dict()
    non_english_rate = float((qualifire["language"] != "en").mean())

    # The key question: among false positives specifically, is non-English text
    # over-represented relative to its base rate in qualifire overall?
    fp = qualifire[qualifire["outcome"] == "fp"]
    fp_non_english_rate = float((fp["language"] != "en").mean()) if len(fp) else None

    benign = qualifire[qualifire["label"] == 0]
    benign_lang_outcome = (
        benign.groupby("language")["outcome"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    report = {
        "n_total": len(qualifire),
        "language_counts": lang_counts,
        "overall_non_english_rate": non_english_rate,
        "n_false_positives": len(fp),
        "fp_non_english_rate": fp_non_english_rate,
        "benign_language_vs_outcome": benign_lang_outcome,
    }
    return report, qualifire

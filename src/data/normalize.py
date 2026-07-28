import pandas as pd

# Necent aggregates many source datasets covering different problems (prompt
# injection, jailbreak, generic toxicity, harmful-content requests, ...). This
# guardrail is scoped to attacks that try to manipulate an LLM's instructions,
# so only these two categories are kept - the rest is dropped.
NECENT_SCOPED_PROMPT_TYPES = ["prompt_injection", "jailbreak"]


def normalize_qualifire(df: pd.DataFrame) -> pd.DataFrame:
    """qualifire/prompt-injections-benchmark: text/label already 1:1, label is a string."""
    return pd.DataFrame({
        "text": df["text"],
        "label": (df["label"] == "jailbreak").astype(int),
        "source_dataset": "qualifire",
        "category": df["label"],
    })


def normalize_neuralchemy(df: pd.DataFrame) -> pd.DataFrame:
    """neuralchemy/Prompt-injection-dataset: text/label already 1:1, label is already 0/1."""
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label"].astype(int),
        "source_dataset": "neuralchemy",
        "category": df["category"],
    })


def normalize_necent(df: pd.DataFrame) -> pd.DataFrame:
    """Necent/llm-jailbreak-prompt-injection-dataset: scope to injection/jailbreak rows.

    prompt_adversarial and is_dangerous are identical within this scope, so
    prompt_adversarial is used as the label: 1 = attack, 0 = labeled-benign
    counterpart (these exist only among the prompt_injection sources).
    """
    scoped = df[df["prompt_type"].isin(NECENT_SCOPED_PROMPT_TYPES)]
    return pd.DataFrame({
        "text": scoped["prompt"],
        "label": scoped["prompt_adversarial"].astype(int),
        "source_dataset": "necent",
        "category": scoped["prompt_type"],
    })


NORMALIZERS = {
    "qualifire/prompt-injections-benchmark": normalize_qualifire,
    "neuralchemy/Prompt-injection-dataset": normalize_neuralchemy,
    "Necent/llm-jailbreak-prompt-injection-dataset": normalize_necent,
}


def build_combined_dataset(raw_dfs: dict) -> pd.DataFrame:
    """Normalize each raw dataset to the common text/label/source_dataset/category
    schema, concatenate them, and drop duplicate/empty text rows."""
    normalized = [NORMALIZERS[name](df) for name, df in raw_dfs.items()]
    combined = pd.concat(normalized, ignore_index=True)

    combined["text"] = combined["text"].str.strip()
    combined = combined[combined["text"] != ""]
    combined = combined.drop_duplicates(subset="text", keep="first")

    return combined.reset_index(drop=True)

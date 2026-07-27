from src.data.load_datasets import load_hf_dataset

DATASET_NAMES = [
    "qualifire/prompt-injections-benchmark",
    "neuralchemy/Prompt-injection-dataset",
    "Necent/llm-jailbreak-prompt-injection-dataset",
]

LABEL_COLUMN_CANDIDATES = ["label", "labels", "category", "class"]


def explore(name, df):
    print(f"\n===== {name} =====")
    print("Shape (rows, columns):", df.shape)
    print("Column names:", df.columns.tolist())

    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
    if label_col:
        print(f"\nLabel distribution ({label_col}):")
        print(df[label_col].value_counts())
    else:
        print(f"\nNo obvious label column found among {LABEL_COLUMN_CANDIDATES} - inspect manually.")

    print("\nFirst 5 rows:")
    print(df.head())


def main():
    for name in DATASET_NAMES:
        try:
            df = load_hf_dataset(name)
        except Exception as error:
            print(f"\n===== {name} =====")
            print(f"Skipped - failed to load: {error}")
            continue
        explore(name, df)


if __name__ == "__main__":
    main()

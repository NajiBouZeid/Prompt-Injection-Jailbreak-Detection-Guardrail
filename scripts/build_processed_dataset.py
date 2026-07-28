import os

from src.config import DATASET_NAMES
from src.data.load_datasets import load_raw_dataset
from src.data.normalize import build_combined_dataset


def main():
    raw_dfs = {name: load_raw_dataset(name) for name in DATASET_NAMES}
    combined = build_combined_dataset(raw_dfs)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/combined.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"Saved {len(combined):,} rows to {out_path}")
    print("\nRows per source dataset:")
    print(combined["source_dataset"].value_counts())
    print("\nLabel distribution (0 = benign, 1 = injection/jailbreak):")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()

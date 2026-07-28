import pandas as pd

from src.data.split import stratified_split


def main():
    combined = pd.read_parquet("data/processed/combined.parquet")
    train_df, val_df, test_df = stratified_split(combined)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_parquet(f"data/processed/{name}.parquet", index=False)
        print(f"{name}: {len(split_df):,} rows")
        print(split_df.groupby("source_dataset")["label"].value_counts(normalize=True))
        print()


if __name__ == "__main__":
    main()

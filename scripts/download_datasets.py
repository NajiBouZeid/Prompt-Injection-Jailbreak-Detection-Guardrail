from src.config import DATASET_NAMES
from src.data.load_datasets import load_hf_dataset, save_raw_dataset


def main():
    for name in DATASET_NAMES:
        try:
            df = load_hf_dataset(name)
        except Exception as error:
            print(f"Skipped {name}: {error}")
            continue

        path = save_raw_dataset(name, df)
        print(f"Saved {name} -> {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

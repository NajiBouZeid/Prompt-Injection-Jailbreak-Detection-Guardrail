import pandas as pd

OUTPUT_PATH = "data/processed/gemini_judge_sample.parquet"
NECENT_SAMPLE_SIZE = 500
RANDOM_SEED = 42


def main():
    df = pd.read_parquet("data/processed/test.parquet")

    # qualifire and neuralchemy are small enough to judge in full (no sampling
    # error there) - they're also where DeBERTa's one known weak spot lives, so
    # full coverage matters more than for necent, which is both huge and already
    # near-perfect for DeBERTa.
    qualifire = df[df["source_dataset"] == "qualifire"]
    neuralchemy = df[df["source_dataset"] == "neuralchemy"]
    necent = df[df["source_dataset"] == "necent"].sample(
        n=NECENT_SAMPLE_SIZE, random_state=RANDOM_SEED
    )

    sample = pd.concat([qualifire, neuralchemy, necent], ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    sample.to_parquet(OUTPUT_PATH)
    print(f"Saved {len(sample)} rows to {OUTPUT_PATH}")
    print(sample["source_dataset"].value_counts())
    print(sample["label"].value_counts())


if __name__ == "__main__":
    main()

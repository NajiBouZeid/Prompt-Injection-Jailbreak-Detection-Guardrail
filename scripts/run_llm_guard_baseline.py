import argparse

from src.evaluation.llm_guard_baseline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", default="data/processed/gemini_judge_sample.parquet")
    parser.add_argument("--output", default="data/processed/llm_guard_results.jsonl")
    args = parser.parse_args()
    run(sample_path=args.sample_path, output_path=args.output)


if __name__ == "__main__":
    main()

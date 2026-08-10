import argparse
import json

from src.evaluation.check_leakage import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/train.parquet")
    parser.add_argument("--test-path", default="data/processed/test.parquet")
    parser.add_argument("--test-sample-size", type=int, default=2000)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="If set, also scores DeBERTa on the sample and breaks down metrics by leaked vs. clean rows.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run(
        train_path=args.train_path,
        test_path=args.test_path,
        test_sample_size=args.test_sample_size,
        model_dir=args.model_dir,
    )

    summary = {k: v for k, v in report.items() if k != "top_embedding_matches"}
    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

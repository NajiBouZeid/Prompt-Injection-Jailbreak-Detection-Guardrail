import argparse
import json

from src.evaluation.evaluate_classifier import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="models/deberta-v3-base-guardrail",
        help="Path to a saved model or checkpoint directory (e.g. .../checkpoint-3000).",
    )
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Truncate the test split for a quick smoke test - leave unset for a real run.",
    )
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument(
        "--output", default=None, help="Optional path to save the report as JSON."
    )
    args = parser.parse_args()

    report = evaluate(
        model_dir=args.model_dir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        latency_samples=args.latency_samples,
    )

    print(json.dumps(report, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

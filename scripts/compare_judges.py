import argparse
import json

from src.evaluation.compare_judges import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default="models/deberta-v3-base-guardrail/checkpoint-23512"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None, help="Optional path to save the report as JSON.")
    args = parser.parse_args()

    report = compare(model_dir=args.model_dir, batch_size=args.batch_size)

    summary = {k: v for k, v in report.items() if k != "disagreements"}
    print(json.dumps(summary, indent=2))
    print(f"\n{len(report['disagreements'])} disagreements (see output file for details)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

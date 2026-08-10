import argparse
import json

from src.evaluation.compare_three_way import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/deberta-v3-base-guardrail/checkpoint-23512")
    parser.add_argument("--output", default="eval_report_three_way_comparison.json")
    args = parser.parse_args()

    report = compare(args.model_dir)
    print(json.dumps({k: v for k, v in report.items() if "source__" not in k}, indent=2))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

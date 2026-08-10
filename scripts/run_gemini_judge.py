import argparse
import json

from src.evaluation.gemini_judge import run_judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument(
        "--output", default="data/processed/gemini_judge_results.jsonl"
    )
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--daily-limit", type=int, default=1000)
    args = parser.parse_args()

    summary = run_judge(
        model=args.model,
        output_path=args.output,
        rpm=args.rpm,
        daily_limit=args.daily_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

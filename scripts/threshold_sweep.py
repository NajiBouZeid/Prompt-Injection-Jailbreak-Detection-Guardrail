import argparse
import json

from src.evaluation.threshold_sweep import best_threshold_for_source, sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/deberta-v3-base-guardrail")
    parser.add_argument("--split", default="val", help="Split to tune the threshold on.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = sweep(model_dir=args.model_dir, split=args.split, batch_size=args.batch_size)

    print(f"\nThreshold sweep on {args.split} ({report['n']} rows):\n")
    header = f"{'thresh':>7} | {'overall F1':>10} | {'qualifire F1':>13} | {'qualifire P':>12} | {'necent F1':>10} | {'neuralchemy F1':>15}"
    print(header)
    print("-" * len(header))
    for entry in report["sweep"]:
        o = entry["overall"]
        q = entry["source__qualifire"]
        n = entry["source__necent"]
        nc = entry["source__neuralchemy"]
        print(
            f"{entry['threshold']:>7.2f} | {o['f1']:>10.4f} | {q['f1']:>13.4f} | "
            f"{q['precision']:>12.4f} | {n['f1']:>10.4f} | {nc['f1']:>15.4f}"
        )

    best = best_threshold_for_source(report, "qualifire", "f1")
    print(f"\nBest threshold for qualifire F1: {best['threshold']} -> {json.dumps(best['source__qualifire'], indent=2)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

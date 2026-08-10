import argparse

from src.inference.guardrail import Guardrail


def main():
    parser = argparse.ArgumentParser(description="Classify a single prompt as attack or benign.")
    parser.add_argument("text")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.model_dir is not None:
        kwargs["model_dir"] = args.model_dir
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold

    guardrail = Guardrail(**kwargs)
    result = guardrail.classify(args.text)
    print(
        f"{result['label'].upper()}  "
        f"(prob_attack={result['prob_attack']:.4f}, threshold={result['threshold']})"
    )


if __name__ == "__main__":
    main()

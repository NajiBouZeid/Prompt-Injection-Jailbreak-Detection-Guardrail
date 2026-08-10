import argparse

from src.training.train_classifier import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir instead of starting fresh.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a specific checkpoint path, overriding --resume's auto-detection "
        "of the latest checkpoint (useful if the latest checkpoint is corrupted).",
    )
    args = parser.parse_args()

    resume = args.resume_from or (args.resume if args.resume else None)
    train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()

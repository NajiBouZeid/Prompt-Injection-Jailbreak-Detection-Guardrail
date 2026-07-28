import argparse

from src.training.train_classifier import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir instead of starting fresh.",
    )
    args = parser.parse_args()

    train(resume_from_checkpoint=args.resume if args.resume else None)


if __name__ == "__main__":
    main()

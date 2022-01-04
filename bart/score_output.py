from pathlib import Path
import argparse
from common import score_rouge
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Evaluate model output in Huggingface Dataset format.")
    parser.add_argument(
        "--output-dir", required=True, help="Path to a model output directory for evaluation."
    )
    parser.add_argument(
        "--pred-column",
        default='pred',
        help="The name of the column containing the predicted model outputs.",
    )
    parser.add_argument(
        "--ref-column",
        default='highlights',
        help="The name of the column containing the reference summaries.",
    )
    args = parser.parse_args()

    outputs = load_from_disk(args.output_dir)
    print(score_rouge(outputs[args.pred_column], outputs[args.ref_column]))


if __name__ == '__main__':
    main()

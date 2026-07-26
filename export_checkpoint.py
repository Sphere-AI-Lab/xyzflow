import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model weights from an XYZFlow training checkpoint")
    parser.add_argument("--input", required=True, help="Training checkpoint path")
    parser.add_argument("--output", required=True, help="Output state_dict path")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        weights = checkpoint
    elif args.weights == "ema" and checkpoint.get("ema") is not None:
        weights = checkpoint["ema"]
    else:
        weights = checkpoint["model"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output_path)
    print(f"Saved {args.weights} weights to {output_path}")


if __name__ == "__main__":
    main()

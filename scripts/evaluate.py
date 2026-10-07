"""Evaluasi ulang sebuah checkpoint pada split test (atau valid) dari manifest mode-nya.

    python scripts/evaluate.py --checkpoint outputs/subset/vgg16/stage_b/best.pt
    python scripts/evaluate.py --checkpoint outputs/subset/vgg16/stage_b/best.pt --split valid
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceauth.data import build_loaders  # noqa: E402
from faceauth.engine import evaluate_checkpoint, load_checkpoint  # noqa: E402
from faceauth.utils import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--out", type=Path, help="folder hasil (default: <folder checkpoint>/<split>_eval)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_checkpoint(args.checkpoint)["config"]
    loaders = build_loaders(cfg, splits=(args.split,))
    out = args.out or args.checkpoint.parent / f"{args.split}_eval"
    evaluate_checkpoint(args.checkpoint, loaders[args.split], loaders["frames"][args.split],
                        loaders["image_root"], out, pick_device(args.device), args.split)
    print(f"Hasil di {out}")


if __name__ == "__main__":
    main()

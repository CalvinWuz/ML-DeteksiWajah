"""Prediksi real/fake untuk gambar baru.

    python scripts/predict.py --checkpoint outputs/subset/vgg16/stage_b/best.pt --images foto.jpg
    python scripts/predict.py --checkpoint outputs/subset/vgg16/stage_b/best.pt --images folder_gambar/ --out hasil.csv

Catatan: model dilatih pada wajah yang sudah dipotong rapi (gaya dataset 140k, real vs StyleGAN).
Gambar di luar gaya itu (bukan wajah, banyak latar, generator lain) bisa menghasilkan prediksi tidak andal.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from faceauth.data import IMG_EXT, build_transforms  # noqa: E402
from faceauth.engine import load_model_from_checkpoint  # noqa: E402
from faceauth.utils import CLASS_NAMES, pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", nargs="+", type=Path, required=True, help="file atau folder gambar")
    ap.add_argument("--threshold", type=float, help="default: threshold di config checkpoint (0.5)")
    ap.add_argument("--out", type=Path, help="simpan hasil ke CSV")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    files = []
    for p in args.images:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    if not files:
        sys.exit("Tidak ada gambar ditemukan.")
    device = pick_device(args.device)
    spec, ck = load_model_from_checkpoint(args.checkpoint, device)
    thr = args.threshold if args.threshold is not None else ck["config"]["train"]["threshold"]
    tf = build_transforms(ck["config"])["eval"]
    rows = []
    with torch.no_grad():
        for f in tqdm(files, desc="Memprediksi", disable=len(files) < 20):
            with Image.open(f) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            prob = torch.sigmoid(spec.model(x)).item()
            rows.append({"file": str(f), "prob_fake": round(prob, 4), "prediksi": CLASS_NAMES[int(prob >= thr)]})
    df = pd.DataFrame(rows)
    print(f"Model {ck['config']['model']['name']} tahap {ck['stage'].upper()} | threshold {thr} | "
          f"label: {ck['label_map']}")
    if len(df) <= 50:
        print(df.to_string(index=False))
    else:
        print(f"{len(df)} gambar diprediksi:\n{df['prediksi'].value_counts().to_string()}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Disimpan ke {args.out}")


if __name__ == "__main__":
    main()

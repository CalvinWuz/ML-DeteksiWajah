"""Training dua tahap + evaluasi test.

  Tahap A (transfer learning): backbone beku, head dilatih.
  Tahap B (fine-tuning)      : muat best.pt tahap A, buka bagian akhir backbone, latih lagi.
  Setelah tiap tahap, best.pt dievaluasi pada split test (tidak dipakai untuk memilih apa pun).

Contoh:
    python scripts/train.py --model vgg16 --mode quick        # pemeriksaan cepat
    python scripts/train.py --model vgg16 --mode subset       # latihan subset 1000+1000
    python scripts/train.py --model vgg16 --mode full         # eksperimen penuh
    python scripts/train.py --model vgg16 --mode subset --resume   # lanjutkan yang terputus
    python scripts/train.py --model vgg16 --mode subset --stage b  # hanya fine-tuning

Hasil: outputs/<run-name>/<model>/stage_a|stage_b/ (checkpoint, history, test/)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from faceauth import metrics as M  # noqa: E402
from faceauth.config import MODELS, MODES, load_config  # noqa: E402
from faceauth.data import build_loaders  # noqa: E402
from faceauth.engine import evaluate_checkpoint, load_checkpoint, train_stage  # noqa: E402
from faceauth.models import build_model, freeze_backbone, summarize, unfreeze_last  # noqa: E402
from faceauth.utils import PROJECT_ROOT, env_info, pick_device, save_json, set_seed  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--mode", required=True, choices=MODES, help="quick | subset | full (wajib dipilih)")
    ap.add_argument("--stage", default="both", choices=["a", "b", "both"])
    ap.add_argument("--run-name", help="nama folder hasil (default = nama mode)")
    ap.add_argument("--resume", action="store_true", help="lanjutkan dari last.pt")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs-a", type=int)
    ap.add_argument("--epochs-b", type=int)
    ap.add_argument("--lr-a", type=float, help="ganti lr tahap A (default: aturan 10%% = 1e-4)")
    ap.add_argument("--lr-b", type=float, help="ganti lr tahap B (default: aturan 10%% = 1e-4)")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--no-test", action="store_true", help="jangan evaluasi test setelah training")
    return ap.parse_args()


def stage_done(stage_dir: Path, epochs: int) -> bool:
    last = stage_dir / "last.pt"
    if not last.exists():
        return False
    ck = load_checkpoint(last)
    patience = ck["config"]["train"]["early_stopping_patience"]
    stopped = patience and ck["epoch"] - ck["best_epoch"] >= patience
    return ck["epoch"] >= epochs or bool(stopped)


def guard(stage_dir: Path, resume: bool):
    if (stage_dir / "last.pt").exists() and not resume:
        sys.exit(f"{stage_dir} sudah berisi hasil. Tambahkan --resume untuk melanjutkan, "
                 "atau pakai --run-name lain agar hasil lama tidak tertimpa.")


def main():
    args = parse()
    ov = {"train": {}, "data": {}}
    for key, val in [("batch_size", args.batch_size), ("epochs_a", args.epochs_a), ("epochs_b", args.epochs_b),
                     ("lr_stage_a", args.lr_a), ("lr_stage_b", args.lr_b)]:
        if val is not None:
            ov["train"][key] = val
    if args.num_workers is not None:
        ov["data"]["num_workers"] = args.num_workers
    cfg = load_config(args.model, args.mode, ov)
    set_seed(cfg["seed"])
    device = pick_device(args.device)
    run_name = args.run_name or args.mode
    out = PROJECT_ROOT / "outputs" / run_name / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    save_json({"env": env_info(), "device": str(device), "command": " ".join(sys.argv)}, out / "run_env.json")

    print(f"Model {args.model} | mode {args.mode} | device {device} | hasil -> {out.relative_to(PROJECT_ROOT)}")
    print(f"Untuk melanjutkan bila terputus:\n  python scripts/train.py --model {args.model} --mode {args.mode} "
          f"--run-name {run_name} --resume\n")
    loaders = build_loaders(cfg)
    fr = loaders["frames"]
    print("Data:", {s: f"{len(fr[s])} ({int(fr[s]['label'].sum())} fake)" for s in fr})
    histories = {}

    # ------------------------------------------------ tahap A
    a_dir = out / "stage_a"
    if args.stage in ("a", "both"):
        skipped = stage_done(a_dir, cfg["train"]["epochs_a"]) and args.resume
        if skipped:
            print("Tahap A sudah selesai; dilewati.")
        else:
            guard(a_dir, args.resume)
            spec = build_model(cfg, pretrained=True)
            report = freeze_backbone(spec)
            spec.model.to(device)
            a_dir.mkdir(parents=True, exist_ok=True)
            (a_dir / "summary.txt").write_text(summarize(spec, loaders["transforms"]["image_size"], device) + "\n")
            print(summarize(spec, loaders["transforms"]["image_size"], device))
            train_stage(spec, cfg, "a", loaders, a_dir, device, args.resume, report)
        if not args.no_test and not (skipped and (a_dir / "test/metrics.json").exists()):
            evaluate_checkpoint(a_dir / "best.pt", loaders["test"], fr["test"], loaders["image_root"],
                                a_dir / "test", device)

    # ------------------------------------------------ tahap B
    b_dir = out / "stage_b"
    if args.stage in ("b", "both"):
        if not (a_dir / "best.pt").exists():
            sys.exit(f"Tahap B butuh {a_dir / 'best.pt'}. Jalankan tahap A dulu.")
        skipped = stage_done(b_dir, cfg["train"]["epochs_b"]) and args.resume
        if skipped:
            print("Tahap B sudah selesai; dilewati.")
        else:
            guard(b_dir, args.resume)
            # Bangun arsitektur tanpa bobot ImageNet, lalu isi dengan bobot terbaik tahap A
            ck = load_checkpoint(a_dir / "best.pt")
            spec = build_model(ck["config"], pretrained=False)
            spec.model.load_state_dict(ck["model_state"])
            u = cfg["model"]["unfreeze"]
            report = unfreeze_last(spec, u["policy"], u["n"])
            print("Kebijakan unfreeze:", report)
            spec.model.to(device)
            b_dir.mkdir(parents=True, exist_ok=True)
            (b_dir / "summary.txt").write_text(summarize(spec, loaders["transforms"]["image_size"], device) + "\n")
            train_stage(spec, cfg, "b", loaders, b_dir, device, args.resume, report,
                        init_from=f"stage_a/best.pt (epoch {ck['epoch']}, val_loss {ck['best_val_loss']:.4f})")
        if not args.no_test and not (skipped and (b_dir / "test/metrics.json").exists()):
            evaluate_checkpoint(b_dir / "best.pt", loaders["test"], fr["test"], loaders["image_root"],
                                b_dir / "test", device)

    # ------------------------------------------------ kurva gabungan
    for st, d in (("A", a_dir), ("B", b_dir)):
        if (d / "last.pt").exists():
            histories[st] = load_checkpoint(d / "last.pt")["history"]
    M.plot_history(histories, out / "curves.png", f"{args.model} ({args.mode})")
    print(f"\nSelesai {time.strftime('%H:%M:%S')}. Kurva: {(out / 'curves.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

"""Data: menemukan dataset, audit, manifest, transform, dan DataLoader.

Manifest = CSV berisi path gambar (relatif ke image_root), label internal, dan split.
Semua model membaca manifest yang sama sehingga split/sampel identik.
"""
import hashlib
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import CLASS_NAMES, LABEL_MAP, PROJECT_ROOT, load_json

SPLITS = ["train", "valid", "test"]
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- menemukan layout
def find_image_root(raw_dir: Path) -> Path:
    """Cari folder yang berisi <split>/{real,fake}/ (mis. real_vs_fake/real-vs-fake)."""
    raw_dir = Path(raw_dir)
    for cand in [raw_dir, *sorted(p for p in raw_dir.rglob("*") if p.is_dir())]:
        if (cand / "train" / "real").is_dir() and (cand / "train" / "fake").is_dir():
            return cand
    raise FileNotFoundError(
        f"Tidak menemukan folder train/real dan train/fake di bawah {raw_dir}. "
        "Pastikan dataset sudah diekstrak."
    )


def scan_folders(image_root: Path) -> pd.DataFrame:
    """Tabel dari struktur folder <split>/<real|fake>/*.jpg (label dari nama folder)."""
    rows = []
    for split_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        split = SPLIT_ALIASES.get(split_dir.name.lower())
        if split is None:
            continue
        for cls in CLASS_NAMES:
            d = split_dir / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"path": f.relative_to(image_root).as_posix(), "split": split, "label_name": cls})
    return pd.DataFrame(rows)


def read_source_csvs(raw_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    """Baca train/valid/test.csv dari sumber Kaggle bila ada. Kembalikan tabel + pemetaan label sumber."""
    frames = []
    for split in SPLITS:
        hits = list(Path(raw_dir).rglob(f"{split}.csv"))
        if hits:
            df = pd.read_csv(hits[0])
            df["split"] = split
            frames.append(df)
    if not frames:
        return None, {}
    df = pd.concat(frames, ignore_index=True)
    # Pemetaan label sumber diturunkan dari data (bukan diasumsikan)
    source_map = {}
    if {"label", "label_str"} <= set(df.columns):
        for name, grp in df.groupby("label_str"):
            vals = sorted(grp["label"].unique().tolist())
            source_map[str(name)] = vals
    return df, source_map


# ---------------------------------------------------------------- audit gambar
def _check_image(args):
    root, rel = args
    p = Path(root) / rel
    try:
        data = p.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(p) as im:
            im.load()  # decode penuh untuk mendeteksi file rusak
            return rel, md5, im.width, im.height, im.mode, ""
    except Exception as e:  # noqa: BLE001 - semua error dicatat sebagai file rusak
        return rel, None, None, None, None, f"{type(e).__name__}: {e}"


def audit_images(image_root: Path, rel_paths: list, workers: int = 8) -> pd.DataFrame:
    from tqdm import tqdm

    out = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap(_check_image, [(str(image_root), p) for p in rel_paths], chunksize=256),
                      total=len(rel_paths), desc="Cek gambar (decode + md5)"):
            out.append(r)
    return pd.DataFrame(out, columns=["path", "md5", "width", "height", "mode", "error"])


def sample_per_class(df: pd.DataFrame, n_per_class, seed: int) -> pd.DataFrame:
    if n_per_class is None:
        return df.copy()
    parts = []
    for label in sorted(df["label"].unique()):
        g = df[df["label"] == label]
        if len(g) < n_per_class:
            raise ValueError(f"Kelas {label} hanya punya {len(g)} gambar, diminta {n_per_class}")
        parts.append(g.sample(n=n_per_class, random_state=seed))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


# ---------------------------------------------------------------- manifest
def manifest_dir(cfg) -> Path:
    return PROJECT_ROOT / cfg["data"]["manifest_dir"]


def load_manifest(cfg) -> tuple[pd.DataFrame, Path]:
    mdir = manifest_dir(cfg)
    path = mdir / f"{cfg['mode']}.csv"
    info_path = mdir / "manifest_info.json"
    if not path.exists() or not info_path.exists():
        raise FileNotFoundError(
            f"Manifest {path} belum ada. Jalankan dulu:\n  python scripts/prepare_data.py --download\n"
            "atau  python scripts/prepare_data.py --source-dir /path/ke/dataset"
        )
    info = load_json(info_path)
    if info["label_map"] != LABEL_MAP:
        raise ValueError(f"Pemetaan label manifest {info['label_map']} != internal {LABEL_MAP}")
    image_root = Path(info["image_root"])
    if not image_root.is_absolute():
        image_root = PROJECT_ROOT / image_root
    return pd.read_csv(path), image_root


# ---------------------------------------------------------------- transform
def build_transforms(cfg) -> dict:
    """Eval/test: transform deterministik bawaan bobot torchvision.
    Train: resize yang sama -> RandomCrop -> RandomHorizontalFlip (augmentasi ringan)."""
    from .models import get_weights

    ref = get_weights(cfg).transforms()  # ImageClassification preset
    crop = ref.crop_size[0]
    resize = ref.resize_size[0]
    normalize = T.Normalize(ref.mean, ref.std)
    train_tf = T.Compose([
        T.Resize(resize, interpolation=ref.interpolation, antialias=True),
        T.RandomCrop(crop),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    desc = {
        "eval": f"Resize({resize}, {ref.interpolation.value}) -> CenterCrop({crop}) -> ToTensor -> Normalize(mean={ref.mean}, std={ref.std})",
        "train": f"Resize({resize}, {ref.interpolation.value}) -> RandomCrop({crop}) -> RandomHorizontalFlip -> ToTensor -> Normalize",
        "image_size": crop,
    }
    return {"train": train_tf, "eval": ref, "desc": desc}


class FaceDataset(Dataset):
    """Membaca gambar satu per satu dari disk (tidak memuat seluruh dataset ke RAM)."""

    def __init__(self, df: pd.DataFrame, image_root: Path, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(self.labels[i]), i


def make_loader(df, image_root, transform, batch_size, shuffle, num_workers, generator=None):
    return DataLoader(
        FaceDataset(df, image_root, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_loaders(cfg, splits=("train", "valid", "test")) -> dict:
    df, image_root = load_manifest(cfg)
    tfs = build_transforms(cfg)
    nw = cfg["data"]["num_workers"]
    out = {"image_root": image_root, "transforms": tfs["desc"], "frames": {}}
    for split in splits:
        part = df[df["split"] == split].reset_index(drop=True)
        out["frames"][split] = part
        if split == "train":
            g = torch.Generator().manual_seed(cfg["seed"])
            out["train_generator"] = g
            out[split] = make_loader(part, image_root, tfs["train"], cfg["train"]["batch_size"], True, nw, g)
        else:
            out[split] = make_loader(part, image_root, tfs["eval"], cfg["train"]["eval_batch_size"], False, nw)
    return out

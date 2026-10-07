"""Susun dataset MODERN: wajah asli vs wajah dari generator lama dan modern.

Sumber (semua dipotong ke wajah 256x256 dengan detektor yang sama, lihat faceauth/faces.py):
  - FFHQ (real) & StyleGAN (fake)         : dataset 140k yang sudah ada
  - IMDB-WIKI (real) & SD 1.5 (fake)      : DeepFakeFace, split per folder identitas
  - Pexels/LAION/... (real) & ~60 generator modern (fake): OpenFake, hasil scripts/fetch_openfake.py
  - gambar Gemini/ChatGPT buatan pengguna : data/user_generated/<app>/  (hanya untuk test)

Setiap potongan disaring CLIP (zero-shot): hanya foto wajah manusia yang dipakai; anime, kartun,
lukisan, patung, dan hewan dibuang. Saringan yang sama berlaku untuk real dan fake.

Grup uji:
  stylegan        FFHQ vs StyleGAN
  dff             IMDB-WIKI vs Stable Diffusion 1.5
  modern_dikenal  real Pexels/LAION vs generator yang JUGA ada di train
  modern_baru     generator yang TIDAK ada di train (mis. gpt-image-1.5/2, nano-banana-pro, midjourney-7)
  lapangan        gambar buatan pengguna dari aplikasi Gemini/ChatGPT

    python scripts/prepare_modern.py
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from faceauth.data import IMG_EXT  # noqa: E402
from faceauth.faces import crop_largest_face  # noqa: E402
from faceauth.utils import LABEL_MAP, PROJECT_ROOT, pick_device, save_json  # noqa: E402

ROOT = PROJECT_ROOT / "data/modern"
MDIR = PROJECT_ROOT / "data/manifests"
SEED = 42
# jumlah per kelas untuk sumber lama (sisanya dari OpenFake)
CAPS = {"stylegan": {"train": 2000, "valid": 200, "test": 500},
        "dff": {"train": 3000, "valid": 300, "test": 800}}
PHOTO_PROMPTS = ["a photo of a real person's face", "a photograph of a human face",
                 "a close-up photo of a person", "a photo of a baby or a young child"]
OTHER_PROMPTS = ["an anime drawing of a face", "a cartoon character", "a painting of a person",
                 "a 3D render of a character", "a sculpture or statue of a face", "a photo of an animal",
                 "a drawing or illustration of a person", "a doll or toy figure", "a video game character"]


def family_of(source: str) -> str:
    s = source.lower()
    rules = [("openai", r"gpt-image|dall"), ("google", r"nano-banana|imagen|gemini"),
             ("midjourney", r"midjourney"), ("flux", r"flux"), ("stylegan", r"stylegan"),
             ("stable-diffusion", r"^sd|stable-diffusion|sdxl|realvis|juggernaut|dreamshaper|epic|realistic_vision|"
                                  r"playground|turbo|touchofrealism|cyberrealistic-xl"),
             ("bytedance", r"seedream"), ("ideogram", r"ideogram"), ("xai", r"grok"),
             ("real", r"ffhq|imdb|pexels|laion|imagenet|docci|real")]
    for fam, pat in rules:
        if re.search(pat, s):
            return fam
    return "lainnya"


# ---------------------------------------------------------------- potong wajah (paralel)
def _crop_job(args):
    src, dst = args
    dst = Path(dst)
    if dst.exists():
        return str(dst), "ada"
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return None, "tidak terbaca"
    crop, info = crop_largest_face(img)
    if crop is None:
        return None, info
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), crop)
    return str(dst), "ok"


def crop_many(pairs, desc, workers=8):
    ok = {}
    with Pool(workers) as p:
        for (src, dst), (res, why) in zip(pairs, tqdm(p.imap(_crop_job, pairs, chunksize=32), total=len(pairs), desc=desc)):
            if res:
                ok[dst] = src
    return ok


# ---------------------------------------------------------------- sumber
def from_140k():
    full = pd.read_csv(MDIR / "full.csv")
    info = json.loads((MDIR / "manifest_info.json").read_text())
    root = PROJECT_ROOT / info["image_root"]
    rows, pairs = [], []
    for split, n in CAPS["stylegan"].items():
        for lab in (0, 1):
            part = full[(full.split == split) & (full.label == lab)].sample(n, random_state=SEED)
            for p in part.path:
                dst = ROOT / "ffhq_stylegan" / split / ("real" if lab == 0 else "fake") / (Path(p).stem + ".png")
                pairs.append((str(root / p), str(dst)))
                rows.append({"dst": str(dst), "label": lab, "split": split, "group": "stylegan",
                             "source": "ffhq" if lab == 0 else "stylegan", "dataset": "140k"})
    ok = crop_many(pairs, "140k")
    return [r for r in rows if r["dst"] in ok]


def from_dff():
    raw = PROJECT_ROOT / "data/raw/deepfakeface"
    for z in ("wiki", "text2img"):
        if not (raw / z).exists() and (raw / f"{z}.zip").exists():
            print(f"Mengekstrak {z}.zip ...")
            with zipfile.ZipFile(raw / f"{z}.zip") as f:
                f.extractall(raw)
    if not (raw / "wiki").exists():
        print("DeepFakeFace belum tersedia; dilewati.")
        return []
    rng = np.random.default_rng(SEED)
    rows, pairs = [], []
    for z, lab, src in (("wiki", 0, "imdb-wiki"), ("text2img", 1, "sd-1.5 (DFF)")):
        files = sorted(p for p in (raw / z).rglob("*") if p.suffix.lower() in IMG_EXT)
        by_split = {"train": [], "valid": [], "test": []}
        for f in files:  # folder 00-99 = identitas; split per folder agar orang yang sama tidak bocor
            d = int(f.relative_to(raw / z).parts[0])
            by_split["train" if d < 80 else "valid" if d < 90 else "test"].append(f)
        for split, fl in by_split.items():
            take = rng.permutation(len(fl))[: int(CAPS["dff"][split] * 1.6)]  # cadangan untuk wajah yang gagal terdeteksi
            for i in take:
                f = fl[i]
                rel = f.relative_to(raw / z)
                dst = ROOT / "dff" / split / ("real" if lab == 0 else "fake") / f"{rel.parts[0]}_{f.stem}.png"
                pairs.append((str(f), str(dst)))
                rows.append({"dst": str(dst), "label": lab, "split": split, "group": "dff", "source": src,
                             "dataset": "DeepFakeFace"})
    ok = crop_many(pairs, "DeepFakeFace")
    df = pd.DataFrame([r for r in rows if r["dst"] in ok])
    parts = []
    for (split, lab), g in df.groupby(["split", "label"]):
        parts.append(g.sample(min(len(g), CAPS["dff"][split]), random_state=SEED))
    return pd.concat(parts).to_dict("records")


def from_openfake():
    rows = []
    sub = {"train": ("train", None), "val": ("valid", None), "test_known": ("test", "modern_dikenal"),
           "test": ("test", "modern_baru")}
    train_models = set()
    tr_idx = ROOT / "openfake/train/index.csv"
    if tr_idx.exists():
        train_models = set(pd.read_csv(tr_idx, on_bad_lines="skip").model)
    for folder, (split, group) in sub.items():
        idx = ROOT / "openfake" / folder / "index.csv"
        if not idx.exists():
            print(f"OpenFake/{folder} belum ada; dilewati.")
            continue
        d = pd.read_csv(idx, on_bad_lines="skip")  # index bisa sedang ditulis
        d = d[[(ROOT / "openfake" / folder / p).exists() for p in d.path]]
        for r in d.itertuples():
            g = group
            if folder == "test":  # generator yang ternyata ada di train dianggap 'dikenal'
                g = "modern_dikenal" if (r.label == "real" or r.model in train_models) else "modern_baru"
            rows.append({"dst": str(ROOT / "openfake" / folder / r.path), "label": LABEL_MAP[r.label],
                         "split": split, "group": g or "modern", "source": r.model, "dataset": "OpenFake"})
    return rows


def from_user():
    rows, pairs = [], []
    base = PROJECT_ROOT / "data/user_generated"
    for app in ("gemini", "chatgpt"):
        for f in sorted((base / app).glob("*")):
            if f.suffix.lower() in IMG_EXT:
                dst = ROOT / "user" / app / (f.stem + ".png")
                pairs.append((str(f), str(dst)))
                rows.append({"dst": str(dst), "label": 1, "split": "test", "group": "lapangan",
                             "source": f"{app} (aplikasi)", "dataset": "buatan pengguna"})
    if not pairs:
        print("Belum ada gambar di data/user_generated/gemini atau chatgpt; grup 'lapangan' kosong.")
        return []
    ok = crop_many(pairs, "gambar pengguna", workers=4)
    return [r for r in rows if r["dst"] in ok]


# ---------------------------------------------------------------- saringan CLIP
def _feat(o):
    """transformers v4 mengembalikan tensor; v5 mengembalikan objek dengan pooler_output (sudah diproyeksikan)."""
    return o if torch.is_tensor(o) else o.pooler_output


@torch.no_grad()
def clip_photo_filter(paths, cache_path: Path, batch=32):
    """Probabilitas 'foto wajah manusia' per gambar (zero-shot CLIP). Hasil di-cache."""
    from transformers import CLIPModel, CLIPProcessor

    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["dst", "p_photo"])
    known = dict(zip(cache.dst, cache.p_photo))
    todo = [p for p in paths if p not in known]
    if todo:
        dev = pick_device()
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        prompts = PHOTO_PROMPTS + OTHER_PROMPTS
        t = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
        tf = _feat(model.get_text_features(**t))
        tf = tf / tf.norm(dim=-1, keepdim=True)
        new = []
        for i in tqdm(range(0, len(todo), batch), desc="Saringan CLIP"):
            chunk = todo[i:i + batch]
            ims = [Image.open(p).convert("RGB") for p in chunk]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            f = _feat(model.get_image_features(pixel_values=px))
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ tf.T).softmax(-1)[:, : len(PHOTO_PROMPTS)].sum(-1).cpu().numpy()
            new += list(zip(chunk, probs))
        cache = pd.concat([cache, pd.DataFrame(new, columns=["dst", "p_photo"])], ignore_index=True)
        cache.to_csv(cache_path, index=False)
        known = dict(zip(cache.dst, cache.p_photo))
    return known


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-photo", type=float, default=0.5, help="ambang probabilitas 'foto wajah' CLIP")
    args = ap.parse_args()

    rows = from_140k() + from_dff() + from_openfake() + from_user()
    df = pd.DataFrame(rows)
    print("Sebelum saringan:", len(df))

    p_photo = clip_photo_filter(df.dst.tolist(), MDIR / "modern_clipfilter.csv")
    df["p_photo"] = df.dst.map(p_photo)
    dropped = df[df.p_photo < args.min_photo]
    df = df[df.p_photo >= args.min_photo].copy()

    df["md5"] = [md5(p) for p in tqdm(df.dst, desc="md5")]
    dup = df[df.duplicated("md5", keep=False)]
    cross = dup.groupby("md5").split.nunique().gt(1).sum()
    df = df.drop_duplicates("md5", keep="first")

    df["path"] = [Path(p).relative_to(ROOT).as_posix() for p in df.dst]
    df["label_name"] = df.label.map({v: k for k, v in LABEL_MAP.items()})
    df["family"] = df.source.map(family_of)
    df.loc[df.split != "test", "group"] = df.loc[df.split != "test", "group"].where(
        df.loc[df.split != "test", "group"].isin(["stylegan", "dff"]), "modern")
    cols = ["path", "label", "label_name", "split", "group", "source", "family", "dataset", "p_photo", "md5"]
    df = df[cols].sort_values(["split", "group", "source", "path"]).reset_index(drop=True)
    MDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MDIR / "modern.csv", index=False)

    # versi kecil untuk pemeriksaan cepat alur
    def take(frame, keys, k):
        return [g.sample(min(len(g), k), random_state=SEED) for _, g in frame.groupby(keys)]
    q = (take(df[df.split == "train"], "label", 48) + take(df[df.split == "valid"], "label", 16)
         + take(df[df.split == "test"], ["group", "label"], 8))
    pd.concat(q).to_csv(MDIR / "modern_quick.csv", index=False)
    for name in ("modern", "modern_quick"):
        save_json({"image_root": "data/modern", "label_map": LABEL_MAP, "positive_class": "fake", "seed": SEED,
                   "description": "Wajah 256x256 hasil potong YuNet; lihat scripts/prepare_modern.py"},
                  MDIR / f"{name}_info.json")

    counts = df.pivot_table(index=["split", "group"], columns="label_name", values="path", aggfunc="count", fill_value=0)
    per_src = df.groupby(["split", "label_name", "family", "source"]).size().rename("n").reset_index()
    per_src.to_csv(MDIR / "modern_per_source.csv", index=False)
    md = ["# Audit dataset modern", "",
          f"- total gambar: {len(df):,}",
          f"- dibuang saringan CLIP (bukan foto wajah manusia, p < {args.min_photo}): {len(dropped):,}",
          f"- duplikat md5 dibuang: {len(dup) - dup.md5.nunique():,} (grup duplikat lintas split: {int(cross)})",
          f"- pemetaan label: {LABEL_MAP}", "", "## Jumlah per split dan grup", "", "```", counts.to_string(), "```",
          "", "## Generator di train vs test", "",
          f"- generator fake di train: {sorted(df[(df.split == 'train') & (df.label == 1)].source.unique())}",
          f"- generator di grup modern_baru (tidak ada di train): "
          f"{sorted(df[df.group == 'modern_baru'].source.unique())}",
          "", "Rincian per sumber: `modern_per_source.csv`."]
    (MDIR / "audit_modern.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

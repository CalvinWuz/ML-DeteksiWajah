"""Unduh/temukan dataset, audit, remap label, buat manifest, dan contoh gambar.

Contoh:
    python scripts/prepare_data.py --download                 # unduh dari Kaggle (~4 GB)
    python scripts/prepare_data.py --source-dir ~/Downloads/140k   # pakai folder yang sudah diunduh

Hasil (data/manifests/):
    full.csv, quick.csv, subset.csv  - manifest tiap mode (path, label, label_name, split)
    manifest_info.json               - image_root, pemetaan label, ringkasan
    audit.json, audit.md             - jumlah per split/kelas, file rusak, duplikasi
    samples.png, distribution.png    - contoh gambar dan distribusi kelas
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from faceauth.data import SPLITS, audit_images, find_image_root, read_source_csvs, sample_per_class, scan_folders  # noqa: E402
from faceauth.utils import CLASS_NAMES, LABEL_MAP, PROJECT_ROOT, save_json  # noqa: E402

KAGGLE_ID = "xhlulu/140k-real-and-fake-faces"


def download_kaggle(raw_dir: Path) -> None:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    if free_gb < 12:
        sys.exit(f"Ruang disk kosong {free_gb:.1f} GB; butuh sekitar 12 GB (zip + ekstraksi).")
    if not (Path.home() / ".kaggle/kaggle.json").exists():
        sys.exit("Tidak ada ~/.kaggle/kaggle.json. Lihat README bagian 'Token Kaggle'.")
    kaggle = Path(sys.executable).parent / "kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "140k-real-and-fake-faces.zip"
    if not zip_path.exists():
        print(f"Mengunduh {KAGGLE_ID} ke {raw_dir} ...")
        subprocess.run([str(kaggle), "datasets", "download", "-d", KAGGLE_ID, "-p", str(raw_dir)], check=True)
    marker = raw_dir / ".extracted"
    if not marker.exists():
        print("Mengekstrak zip ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        marker.touch()


def build_table(raw_dir: Path, workers: int, skip_decode: bool):
    image_root = find_image_root(raw_dir)
    print(f"image_root: {image_root}")
    folders = scan_folders(image_root)
    csv_df, source_map = read_source_csvs(raw_dir)
    notes = []

    if csv_df is not None:
        # CSV Kaggle: kolom 'path' relatif ke image_root, 'label' angka, 'label_str' nama kelas
        csv_df = csv_df.rename(columns={"label": "source_label"})
        csv_df["label_name"] = csv_df["label_str"].str.lower()
        notes.append(f"Pemetaan label SUMBER (dari CSV): {source_map}")
        # Label dari CSV harus cocok dengan nama folder
        folder_cls = csv_df["path"].str.split("/").str[1].str.lower()
        mismatch = int((folder_cls != csv_df["label_name"]).sum())
        notes.append(f"Label CSV vs nama folder tidak cocok: {mismatch}")
        csv_split = csv_df["path"].str.split("/").str[0].map(lambda s: {"valid": "valid"}.get(s, s))
        notes.append(f"Split CSV vs folder tidak cocok: {int((csv_split != csv_df['split']).sum())}")
        missing = sorted(set(csv_df["path"]) - set(folders["path"]))
        extra = sorted(set(folders["path"]) - set(csv_df["path"]))
        notes.append(f"Ada di CSV tapi tidak ada filenya: {len(missing)} {missing[:5]}; "
                     f"file tanpa baris CSV (tidak dipakai): {len(extra)} {extra[:5]}")
        table = csv_df[["path", "split", "label_name", "source_label"]].copy()
        table = table[~table["path"].isin(missing)]
    else:
        notes.append("CSV tidak ditemukan: label diambil dari nama folder real/fake.")
        table = folders.copy()
        table["source_label"] = None

    table["label"] = table["label_name"].map(LABEL_MAP)  # remap ke internal real=0, fake=1
    if table["label"].isna().any():
        raise ValueError(f"Ada label di luar {list(LABEL_MAP)}: {table[table['label'].isna()]['label_name'].unique()}")
    table["label"] = table["label"].astype(int)

    if skip_decode:
        notes.append("Pemeriksaan decode/md5 DILEWATI (--skip-decode-check): duplikasi tidak diperiksa.")
        table["md5"] = None
        bad = pd.DataFrame(columns=["path", "error"])
    else:
        aud = audit_images(image_root, table["path"].tolist(), workers)
        table = table.merge(aud, on="path", how="left")
        bad = table[table["error"].fillna("") != ""][["path", "error"]]
        table = table[table["error"].fillna("") == ""].drop(columns=["error"])
    return image_root, table, bad, notes


def duplicate_report(table: pd.DataFrame) -> dict:
    if table["md5"].isna().all():
        return {"checked": False}
    groups = table.groupby("md5")
    dup = groups.filter(lambda g: len(g) > 1)
    cross = groups.filter(lambda g: g["split"].nunique() > 1)
    conflict = groups.filter(lambda g: g["label"].nunique() > 1)
    return {
        "checked": True,
        "method": "md5 isi file (duplikat persis; near-duplicate tidak diperiksa)",
        "files_in_duplicate_groups": int(len(dup)),
        "duplicate_groups": int(dup["md5"].nunique()),
        "files_duplicated_across_splits": int(len(cross)),
        "groups_with_conflicting_labels": int(conflict["md5"].nunique()) if len(conflict) else 0,
        "examples_across_splits": cross.sort_values("md5").head(10)[["md5", "path", "split"]].to_dict("records"),
    }


def plot_samples(table, image_root, path, seed):
    fig, axes = plt.subplots(2, 6, figsize=(12, 5.4))
    for r, cls in enumerate(CLASS_NAMES):
        part = table[(table["label_name"] == cls) & (table["split"] == "train")].sample(6, random_state=seed)
        for c, (_, row) in enumerate(part.iterrows()):
            with Image.open(image_root / row["path"]) as im:
                axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].set_title(f"{cls} (label {row['label']})\n{Path(row['path']).name[:14]}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("Contoh gambar train: baris atas real (0), baris bawah fake (1)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_distribution(counts: pd.DataFrame, path):
    ax = counts.plot(kind="bar", figsize=(6, 3.6), rot=0)
    ax.set_ylabel("jumlah gambar")
    ax.set_title("Distribusi kelas per split (dataset lengkap)")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="{:,.0f}", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="unduh dataset dari Kaggle")
    ap.add_argument("--source-dir", type=Path, help="folder dataset yang sudah diunduh/diekstrak")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-decode-check", action="store_true", help="lewati decode + md5 (lebih cepat, audit berkurang)")
    args = ap.parse_args()

    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text())
    seed = base["seed"]
    raw_dir = args.source_dir.expanduser().resolve() if args.source_dir else PROJECT_ROOT / base["data"]["raw_dir"]
    if args.download:
        download_kaggle(raw_dir)
    if not raw_dir.exists():
        sys.exit(f"Folder dataset {raw_dir} tidak ada. Pakai --download atau --source-dir.")

    image_root, table, bad, notes = build_table(raw_dir, args.workers, args.skip_decode_check)
    out = PROJECT_ROOT / base["data"]["manifest_dir"]
    out.mkdir(parents=True, exist_ok=True)

    counts = table.pivot_table(index="split", columns="label_name", values="path", aggfunc="count").reindex(SPLITS)[CLASS_NAMES]
    dups = duplicate_report(table)
    cols = ["path", "label", "label_name", "split", "source_label", "md5", "width", "height", "mode"]
    full = table[[c for c in cols if c in table.columns]].sort_values(["split", "path"]).reset_index(drop=True)
    full.to_csv(out / "full.csv", index=False)

    mode_counts = {"full": counts.to_dict()}
    for mode in ("quick", "subset"):
        mc = base["modes"][mode]
        parts = [sample_per_class(full[full["split"] == s], mc[f"{k}_per_class"], seed)
                 for s, k in (("train", "train"), ("valid", "val"), ("test", "test"))]
        m = pd.concat(parts, ignore_index=True)
        m.to_csv(out / f"{mode}.csv", index=False)
        mode_counts[mode] = m.pivot_table(index="split", columns="label_name", values="path", aggfunc="count").to_dict()

    try:
        rel_root = image_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_root = str(image_root)
    info = {
        "source": base["data"]["source"] if args.download or args.source_dir is None else f"folder lokal: {raw_dir}",
        "image_root": rel_root,
        "label_map": LABEL_MAP,
        "positive_class": "fake",
        "seed": seed,
        "counts_per_mode": mode_counts,
    }
    save_json(info, out / "manifest_info.json")

    sizes = table.groupby(["width", "height"]).size().to_dict() if "width" in table else {}
    audit = {
        "image_root": str(image_root),
        "n_valid_images": int(len(table)),
        "n_bad_images": int(len(bad)),
        "bad_images": bad.head(50).to_dict("records"),
        "counts": counts.to_dict(),
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in sizes.items()},
        "duplicates": dups,
        "notes": notes,
        "label_map_internal": LABEL_MAP,
    }
    save_json(audit, out / "audit.json")
    md = ["# Audit dataset", "", f"- image_root: `{image_root}`", f"- gambar valid: {len(table):,}",
          f"- gambar rusak/tidak terbaca: {len(bad):,}", f"- pemetaan internal: {LABEL_MAP} (positif = fake)", ""]
    md += [f"- {n}" for n in notes]
    md += ["", "## Jumlah per split", "", "```", counts.to_string(), "```", "",
           "## Ukuran gambar", ""] + [f"- {k}: {v:,}" for k, v in audit["image_sizes"].items()]
    md += ["", "## Duplikasi", "", "```", str({k: v for k, v in dups.items() if k != "examples_across_splits"}), "```"]
    for mode in ("quick", "subset"):
        md += ["", f"## Manifest mode `{mode}`", "", "```", str(pd.DataFrame(mode_counts[mode])), "```"]
    (out / "audit.md").write_text("\n".join(md) + "\n")

    plot_samples(table, image_root, out / "samples.png", seed)
    plot_distribution(counts, out / "distribution.png")

    print("\n".join(notes))
    print("\nJumlah gambar per split:\n", counts, sep="")
    print(f"\nGambar rusak: {len(bad)} | duplikasi: {({k: v for k, v in dups.items() if k != 'examples_across_splits'})}")
    for mode in ("quick", "subset"):
        print(f"\nMode {mode}:\n{pd.DataFrame(mode_counts[mode])}")
    print(f"\nSelesai. Lihat {out.relative_to(PROJECT_ROOT)}/audit.md dan samples.png")


if __name__ == "__main__":
    main()

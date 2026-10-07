"""Ambil wajah dari OpenFake (generator modern) tanpa mengunduh file parquet utuh.

OpenFake (ComplexDataLab, CC-BY-NC-4.0) berisi gambar asli dan gambar dari generator modern,
termasuk gpt-image (ChatGPT), nano-banana / imagen (Gemini/Google), midjourney, flux,
seedream, dll. Skrip ini:
  1. membaca kolom metadata (model, label, prompt) sebuah shard,
  2. memilih baris yang prompt-nya menggambarkan orang dan bukan ilustrasi/poster,
  3. mengunduh hanya halaman parquet berisi baris tersebut,
  4. mendeteksi wajah terbesar dan menyimpan potongan 256x256 PNG.

    python scripts/fetch_openfake.py --split test --shards 0 1
    python scripts/fetch_openfake.py --split train --shards 0 5 10 --cap-fake 1500

Hasil: data/modern/openfake/<split>/<real|fake>/<model>/<row_id>.png + index.csv
Bisa dihentikan dan dijalankan ulang: row group yang sudah selesai dilewati.
"""
import argparse
import csv
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

from faceauth.faces import crop_largest_face, decode_image, save_png  # noqa: E402
from faceauth.parquet_pages import RemoteFile, fetch_rows  # noqa: E402
from faceauth.utils import PROJECT_ROOT  # noqa: E402

REPO = "ComplexDataLab/OpenFake"
# Semua generator gambar dipakai KECUALI yang menghasilkan anime/kartun atau frame video,
# karena tugasnya wajah fotorealistis.
EXCLUDE_MODELS = re.compile(r"anime|animagine|pony|ghibli|illustrious|noobai|anima-|novaanime|"
                            r"video|veo|sora|wan2|wan-|hunyuan|ltxv|tiny-random|halfmoon|frames-|aurora-|lumina-", re.I)
PERSON = re.compile(r"\b(man|men|woman|women|person|people|girl|boy|child|children|kid|lady|guy|couple|"
                    r"individuals?|portrait|face|selfie|bride|groom|actor|actress|teen(ager)?|baby|toddler|"
                    r"elderly|student|worker|player|he|she|his|her)\b", re.I)
NOT_PHOTO = re.compile(r"\b(poster|cartoon|illustration|anime|manga|painting|drawing|sketch|logo|cover|comic|"
                       r"render(ing)?|3d|statue|sculpture|doll|figurine|text overlay|meme|screenshot|infographic|"
                       r"diagram|vector|clip ?art|emoji|mascot|toy|lego|chart|map)\b", re.I)
COLS = ["model", "label", "prompt", "row_id"]
lock = threading.Lock()


def shard_files(fs, split):
    return sorted(p for p in fs.ls(f"datasets/{REPO}/core", detail=False) if Path(p).name.startswith(f"{split}-"))


def process_rg(path, pf_meta, img_col, max_def, rg, rows, out_dir, writer, counts, caps, done_f):
    rf = RemoteFile(f"https://huggingface.co/datasets/{REPO}/resolve/main/{path.split(REPO + '/')[1]}")
    with lock:  # batasi kuota per model sebelum mengunduh
        rows = [r for r in rows if counts[(r["label"], r["model"])] < caps[r["label"]]]
    if rows:
        col = pf_meta.row_group(rg).column(img_col)
        start = col.dictionary_page_offset or col.data_page_offset
        got = fetch_rows(rf, start, col.total_compressed_size, {r["i"] for r in rows}, max_def)
        for r in rows:
            data = got.get(r["i"])
            if not data:
                continue
            img = decode_image(data)
            if img is None:
                continue
            crop, info = crop_largest_face(img)
            if crop is None:
                continue
            with lock:
                key = (r["label"], r["model"])
                if counts[key] >= caps[r["label"]]:
                    continue
                counts[key] += 1
            rel = Path(r["label"]) / r["model"] / f"{r['row_id']}.png"
            save_png(crop, out_dir / rel)
            with lock:
                writer.writerow({"path": rel.as_posix(), "label": r["label"], "model": r["model"],
                                 "row_id": r["row_id"], "shard": Path(path).name, **info,
                                 "prompt": r["prompt"][:300]})
    with lock:
        done_f.write(f"{Path(path).name}:{rg}\n")
        done_f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--shards", nargs="+", type=int, required=True, help="indeks file shard dalam split")
    ap.add_argument("--out-name", help="nama folder hasil (default = nama split)")
    ap.add_argument("--cap-fake", type=int, default=1500, help="maksimal wajah per generator")
    ap.add_argument("--cap-real", type=int, default=5000, help="maksimal wajah asli per sumber")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "data/modern/openfake" / (args.out_name or args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, done_path = out_dir / "index.csv", out_dir / ".done"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    counts = Counter()
    if index_path.exists():
        import pandas as pd
        for (lab, mod), n in pd.read_csv(index_path).groupby(["label", "model"]).size().items():
            counts[(lab, mod)] = n
    caps = {"fake": args.cap_fake, "real": args.cap_real}

    fs = HfFileSystem()
    files = shard_files(fs, args.split)
    print(f"{len(files)} shard tersedia untuk split {args.split}")
    new_file = not index_path.exists()
    with open(index_path, "a", newline="") as fidx, open(done_path, "a") as done_f:
        writer = csv.DictWriter(fidx, fieldnames=["path", "label", "model", "row_id", "shard", "n_faces",
                                                  "face_px", "face_score", "src_w", "src_h", "prompt"])
        if new_file:
            writer.writeheader()
        for si in args.shards:
            path = files[si]
            with fs.open(path, block_size=2 ** 20) as f:
                pf = pq.ParquetFile(f)
                meta = pf.metadata
                names = [meta.schema.column(i).path for i in range(meta.num_columns)]
                img_col = names.index("image.bytes")
                max_def = meta.schema.column(img_col).max_definition_level
                todo = [rg for rg in range(meta.num_row_groups) if f"{Path(path).name}:{rg}" not in done]
                jobs = []
                for rg in todo:
                    t = pf.read_row_group(rg, columns=COLS).to_pylist()
                    rows = [{**r, "i": i} for i, r in enumerate(t)
                            if (r["label"] == "real" or not EXCLUDE_MODELS.search(r["model"]))
                            and PERSON.search(r["prompt"] or "") and not NOT_PHOTO.search(r["prompt"] or "")]
                    jobs.append((rg, rows))
            print(f"[{Path(path).name}] {len(todo)} row group, kandidat: "
                  f"{Counter(r['model'] for _, rows in jobs for r in rows).most_common(12)}")
            with ThreadPoolExecutor(args.workers) as ex:
                futs = [ex.submit(process_rg, path, meta, img_col, max_def, rg, rows, out_dir, writer,
                                  counts, caps, done_f) for rg, rows in jobs]
                for k, fu in enumerate(as_completed(futs), 1):
                    fu.result()
                    if k % 8 == 0 or k == len(futs):
                        print(f"  {k}/{len(futs)} row group | wajah tersimpan: "
                              f"{sum(counts.values())} {dict(sorted((f'{a}/{b}', n) for (a, b), n in counts.items()))}",
                              flush=True)


if __name__ == "__main__":
    main()

"""Tabel perbandingan 3 model x 2 tahap untuk satu run.

    python scripts/compare.py --run-name subset

Hasil: outputs/<run-name>/comparison.csv dan comparison.md.
Kombinasi yang belum dijalankan ditandai "belum tersedia".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from faceauth.config import BRIEF_MODELS, MODELS  # noqa: E402
from faceauth.utils import PROJECT_ROOT, load_json  # noqa: E402

NA = "belum tersedia"


def row_for(run_dir: Path, model: str, stage: str) -> dict:
    d = run_dir / model / f"stage_{stage}"
    row = {"model": model, "tahap": "A transfer learning" if stage == "a" else "B fine-tuning"}
    mpath, ipath = d / "test/metrics.json", d / "train_info.json"
    if not (mpath.exists() and ipath.exists()):
        return {**row, "status": NA}
    m, i = load_json(mpath), load_json(ipath)
    pc = m["per_class"]
    return {
        **row, "status": "terukur",
        "accuracy": round(m["accuracy"], 4), "roc_auc": None if m["roc_auc"] is None else round(m["roc_auc"], 4),
        "avg_precision": None if m["average_precision"] is None else round(m["average_precision"], 4),
        "f1_real": round(pc["real"]["f1"], 4), "f1_fake": round(pc["fake"]["f1"], 4),
        "precision_fake": round(pc["fake"]["precision"], 4), "recall_fake": round(pc["fake"]["recall"], 4),
        "macro_f1": round(m["macro_f1"], 4), "n_test": m["n"], "best_epoch": i.get("best_epoch"),
        "epochs_run": i.get("epochs_run"), "lr": i["lr"], "batch_size": i["batch_size"],
        "params_total": i["params"]["total"], "params_trainable": i["params"]["trainable"],
        "unfreeze": (f"{i['unfreeze']['policy']} n={i['unfreeze']['n']}" if stage == "b" else "head saja"),
        "n_train": i["n_train"], "n_val": i["n_val"], "mode": i["mode"], "seed": i["seed"],
        "device": i["device"], "train_min": round(i.get("train_duration_sec", 0) / 60, 1),
        # dataset modern: tingkat deteksi fake per grup uji + proporsi real yang benar
        **{f"deteksi_{g}": round(v["recall_fake"], 4) for g, v in m.get("per_group", {}).items() if v["n_fake"]},
        **({"real_benar": round(pc["real"]["recall"], 4)} if m.get("per_group") else {}),
    }


def plot_roc_all(run_dir: Path, name: str) -> Path | None:
    """Kurva ROC semua model x tahap yang sudah punya predictions.csv, dalam satu grafik."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    found = False
    for model in MODELS:
        for stage in ("a", "b"):
            f = run_dir / model / f"stage_{stage}" / "test" / "predictions.csv"
            if not f.exists():
                continue
            found = True
            d = pd.read_csv(f)
            fpr, tpr, _ = roc_curve(d["label"], d["prob_fake"], pos_label=1)
            auc = roc_auc_score(d["label"], d["prob_fake"])
            for ax in axes:
                ax.plot(fpr, tpr, ls="--" if stage == "a" else "-", label=f"{model} {stage.upper()} (AUC {auc:.4f})")
    if not found:
        plt.close(fig)
        return None
    axes[1].set_xlim(0, 0.05)
    axes[1].set_ylim(0.9, 1.001)
    for ax, title in zip(axes, ["Kurva ROC (test)", "Perbesaran: FPR 0-5%"]):
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
        ax.set_xlabel("False positive rate (real ditebak fake)")
        ax.set_ylabel("True positive rate (fake terdeteksi)")
        ax.set_title(title)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Run {name}: garis putus = tahap A, garis penuh = tahap B")
    fig.tight_layout()
    out = run_dir / "roc_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    run_dir = PROJECT_ROOT / "outputs" / args.run_name
    # model brief selalu ditampilkan; CLIP (perluasan) hanya jika run ini memang melatihnya
    models = [m for m in MODELS if m in BRIEF_MODELS or (run_dir / m).exists()]
    df = pd.DataFrame([row_for(run_dir, m, s) for m in models for s in ("a", "b")])
    for c in ("best_epoch", "epochs_run", "batch_size", "params_total", "params_trainable", "n_train", "n_val", "n_test", "seed"):
        if c in df:
            df[c] = df[c].astype("Int64")  # tetap bilangan bulat walau ada baris "belum tersedia"
    df.to_csv(run_dir / "comparison.csv", index=False)
    cols = ["model", "tahap", "status", "accuracy", "roc_auc", "macro_f1", "f1_real", "f1_fake",
            "best_epoch", "epochs_run", "lr", "unfreeze", "params_trainable", "n_train", "device", "train_min"]
    cols += ["real_benar"] + sorted(c for c in df.columns if c.startswith("deteksi_"))
    show = df[[c for c in cols if c in df.columns]].astype(object).fillna("-")
    lines = [f"# Perbandingan run `{args.run_name}`", "",
             "Metrik pada split test, threshold 0.5, kelas positif = fake. "
             "Checkpoint dipilih dengan val_loss minimum.", "",
             "| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in show.values]
    (run_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(show.to_string(index=False))
    print(f"\nDisimpan: {run_dir.relative_to(PROJECT_ROOT)}/comparison.csv dan comparison.md")
    roc = plot_roc_all(run_dir, args.run_name)
    if roc:
        print(f"Grafik ROC gabungan: {roc.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

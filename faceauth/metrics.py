"""Metrik evaluasi dan visualisasi. ROC/AUC selalu dari prob_fake (kelas positif = fake = 1)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

from .utils import CLASS_NAMES  # noqa: E402


def compute_metrics(y_true, prob_fake, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob_fake = np.asarray(prob_fake, dtype=float)
    y_pred = (prob_fake >= threshold).astype(int)
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {
        "n": int(len(y_true)),
        "threshold": threshold,
        "positive_class": "fake (label 1)",
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(2)
        },
        "macro_f1": float(np.mean(f)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_note": "baris = label asli [real, fake], kolom = prediksi [real, fake]",
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(skm.roc_auc_score(y_true, prob_fake))
        out["average_precision"] = float(skm.average_precision_score(y_true, prob_fake))
    else:
        out["roc_auc"] = out["average_precision"] = None
    return out


def plot_confusion(cm, path, title=""):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], [f"pred {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"asli {c}" for c in CLASS_NAMES])
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_pr(y_true, prob_fake, out_dir: Path, title=""):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = skm.roc_curve(y_true, prob_fake, pos_label=1)
    auc = skm.roc_auc_score(y_true, prob_fake)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate (real diprediksi fake)")
    ax.set_ylabel("True positive rate (fake terdeteksi)")
    ax.set_title(title or "Kurva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=130)
    plt.close(fig)

    prec, rec, _ = skm.precision_recall_curve(y_true, prob_fake, pos_label=1)
    ap = skm.average_precision_score(y_true, prob_fake)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(rec, prec, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall (fake)")
    ax.set_ylabel("Precision (fake)")
    ax.set_title((title + " - " if title else "") + "Precision-Recall (pelengkap)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=130)
    plt.close(fig)


def plot_history(histories: dict, path: Path, title=""):
    """histories = {"A": [rows], "B": [rows]}; garis vertikal menandai batas tahap."""
    rows, boundaries, offset = [], [], 0
    for stage, hist in histories.items():
        for h in hist:
            rows.append({**h, "global_epoch": offset + h["epoch"], "stage": stage})
        offset += len(hist)
        boundaries.append((offset, stage))
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axes, ["loss", "acc"]):
        ax.plot(df["global_epoch"], df[f"train_{key}"], "o-", label=f"train {key}")
        ax.plot(df["global_epoch"], df[f"val_{key}"], "s-", label=f"val {key}")
        for b, stage in boundaries[:-1]:
            ax.axvline(b + 0.5, color="red", ls="--", lw=1)
            ax.text(b + 0.5, 0.02, " A | B ", color="red", ha="center", va="bottom", fontsize=8,
                    transform=ax.get_xaxis_transform(), backgroundcolor="white")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("epoch gabungan (garis merah = mulai fine-tuning)")
        ax.set_title(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_examples(pred_df: pd.DataFrame, image_root: Path, path: Path, n=6, seed=0, title=""):
    """Contoh prediksi benar dan salah beserta prob_fake."""
    correct = pred_df[pred_df["label"] == pred_df["pred"]]
    wrong = pred_df[pred_df["label"] != pred_df["pred"]]
    picks = [("BENAR", correct.sample(min(n, len(correct)), random_state=seed)),
             ("SALAH", wrong.sample(min(n, len(wrong)), random_state=seed))]
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.8))
    for r, (tag, part) in enumerate(picks):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(part):
                row = part.iloc[c]
                with Image.open(Path(image_root) / row["path"]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{tag}\nasli={CLASS_NAMES[row['label']]} pred={CLASS_NAMES[row['pred']]}\n"
                             f"p(fake)={row['prob_fake']:.3f}", fontsize=7,
                             color="green" if tag == "BENAR" else "red")
            elif c == 0:
                ax.text(0.5, 0.5, f"tidak ada prediksi {tag.lower()}", ha="center", va="center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

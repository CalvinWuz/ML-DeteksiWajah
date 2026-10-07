"""Loop training/evaluasi, checkpoint, dan resume."""
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from . import metrics as M
from .models import ModelSpec, build_model, set_train_mode, trainable_names
from .utils import CLASS_NAMES, LABEL_MAP, count_params, save_json


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(path: Path, **state) -> None:
    """Tulis ke file sementara lalu rename, agar checkpoint tidak rusak bila terputus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)  # hanya tensor & tipe dasar


def load_model_from_checkpoint(path: Path, device) -> tuple[ModelSpec, dict]:
    ckpt = load_checkpoint(path)
    spec = build_model(ckpt["config"], pretrained=False)
    spec.model.load_state_dict(ckpt["model_state"])
    spec.model.to(device).eval()
    return spec, ckpt


def _rng_state() -> dict:
    """State RNG dalam tipe dasar agar aman dimuat dengan weights_only=True."""
    kind, keys, pos, has_gauss, gauss = np.random.get_state()
    py = random.getstate()
    return {"torch": torch.get_rng_state(),
            "numpy": [kind, keys.tolist(), int(pos), int(has_gauss), float(gauss)],
            "python": [py[0], list(py[1]), py[2]]}


# ---------------------------------------------------------------- satu epoch
def run_epoch(spec: ModelSpec, loader, device, criterion, optimizer=None, batchnorm="eval", desc=""):
    train = optimizer is not None
    if train:
        set_train_mode(spec, batchnorm)
    else:
        spec.model.eval()
    total_loss, n, probs, labels, idxs = 0.0, 0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y, idx in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0):
            x, y = x.to(device), y.to(device)
            logits = spec.model(x).squeeze(1)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logits).detach().float().cpu())
            labels.append(y.detach().cpu())
            idxs.append(idx)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy().astype(int)
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    auc = M.compute_metrics(labels, probs)["roc_auc"]
    return {"loss": total_loss / n, "acc": acc, "auc": auc, "n": n,
            "probs": probs, "labels": labels, "idx": torch.cat(idxs).numpy()}


# ---------------------------------------------------------------- satu tahap
def train_stage(spec: ModelSpec, cfg: dict, stage: str, loaders: dict, out_dir: Path,
                device, resume: bool, unfreeze_report: dict, init_from: str | None = None) -> dict:
    """stage = 'a' (transfer learning) atau 'b' (fine-tuning)."""
    t = cfg["train"]
    epochs = t[f"epochs_{stage}"]
    lr = t[f"lr_stage_{stage}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer dibangun ulang dan hanya berisi parameter yang trainable pada tahap ini
    params = [p for p in spec.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=t["weight_decay"])

    history, best_loss, best_epoch, start_epoch, elapsed_prev = [], float("inf"), 0, 1, 0.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.exists():
        ck = load_checkpoint(last_path)
        spec.model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        history, best_loss, best_epoch = ck["history"], ck["best_val_loss"], ck["best_epoch"]
        start_epoch, elapsed_prev = ck["epoch"] + 1, ck.get("elapsed_sec", 0.0)
        torch.set_rng_state(ck["rng"]["torch"])
        kind, keys, pos, has_gauss, gauss = ck["rng"]["numpy"]
        np.random.set_state((kind, np.array(keys, dtype=np.uint32), pos, has_gauss, gauss))
        py = ck["rng"]["python"]
        random.setstate((py[0], tuple(py[1]), py[2]))
        print(f"[resume] melanjutkan tahap {stage.upper()} dari epoch {start_epoch} (best epoch {best_epoch})")

    info = {
        "stage": stage, "mode": cfg["mode"], "model": cfg["model"]["name"],
        "data_subset": cfg["data"]["subset"], "seed": cfg["seed"],
        "n_train": len(loaders["frames"]["train"]), "n_val": len(loaders["frames"]["valid"]),
        "epochs_planned": epochs, "lr": lr, "lr_source": t[f"lr_stage_{stage}_source"],
        "optimizer": "Adam", "batch_size": t["batch_size"], "scheduler": t["scheduler"],
        "early_stopping_patience": t["early_stopping_patience"],
        "checkpoint_criterion": "val_loss minimum", "batchnorm": t["batchnorm"],
        "device": str(device), "params": count_params(spec.model), "unfreeze": unfreeze_report,
        "init_from": init_from or f"bobot pretrained {cfg['model']['weights']}",
        "transforms": loaders["transforms"], "label_map": LABEL_MAP,
    }
    save_json(info, out_dir / "train_info.json")
    (out_dir / "trainable_params.txt").write_text("\n".join(trainable_names(spec.model)) + "\n")

    t0 = time.time()
    patience = t["early_stopping_patience"]
    for epoch in range(start_epoch, epochs + 1):
        loaders["train_generator"].manual_seed(cfg["seed"] + epoch)  # urutan acak per epoch tetap
        ep_t0 = time.time()
        tr = run_epoch(spec, loaders["train"], device, criterion, optimizer, t["batchnorm"],
                       desc=f"{stage.upper()} ep{epoch} train")
        va = run_epoch(spec, loaders["valid"], device, criterion, desc=f"{stage.upper()} ep{epoch} val")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_acc": tr["acc"], "train_auc": tr["auc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_auc": va["auc"],
               "epoch_sec": round(time.time() - ep_t0, 1)}
        history.append(row)
        improved = va["loss"] < best_loss
        if improved:
            best_loss, best_epoch = va["loss"], epoch
        state = dict(model_state=spec.model.state_dict(), config=cfg, stage=stage, epoch=epoch,
                     history=history, best_val_loss=best_loss, best_epoch=best_epoch,
                     label_map=LABEL_MAP, class_names=CLASS_NAMES,
                     trainable_params=trainable_names(spec.model), unfreeze=unfreeze_report,
                     elapsed_sec=elapsed_prev + time.time() - t0)
        if improved:
            save_checkpoint(best_path, **state)
        save_checkpoint(last_path, **state, optimizer_state=optimizer.state_dict(),
                        rng=_rng_state())
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"[{stage.upper()}] epoch {epoch}/{epochs} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} auc {va['auc'] if va['auc'] is None else round(va['auc'], 4)} | "
              f"{row['epoch_sec']}s{'  *best*' if improved else ''}")
        if patience and epoch - best_epoch >= patience:
            print(f"[{stage.upper()}] early stopping: val_loss tidak membaik {patience} epoch")
            break

    info.update({"epochs_run": len(history), "best_epoch": best_epoch, "best_val_loss": best_loss,
                 "train_duration_sec": round(elapsed_prev + time.time() - t0, 1)})
    save_json(info, out_dir / "train_info.json")
    return {"history": history, "best_path": best_path, "info": info}


# ---------------------------------------------------------------- evaluasi checkpoint
@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path, loader, frame: pd.DataFrame, image_root: Path,
                        out_dir: Path, device, split_name="test") -> dict:
    """Evaluasi checkpoint pada satu split; simpan metrik, prediksi, dan grafik."""
    spec, ckpt = load_model_from_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    if ckpt["label_map"] != LABEL_MAP:
        raise ValueError(f"label_map checkpoint {ckpt['label_map']} != {LABEL_MAP}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_epoch(spec, loader, device, nn.BCEWithLogitsLoss(), desc=f"eval {split_name}")
    thr = cfg["train"]["threshold"]
    m = M.compute_metrics(res["labels"], res["probs"], thr)
    m.update({"split": split_name, "loss": res["loss"], "checkpoint": str(ckpt_path),
              "checkpoint_epoch": ckpt["epoch"], "stage": ckpt["stage"], "model": cfg["model"]["name"],
              "mode": cfg["mode"], "data_subset": cfg["data"]["subset"], "eval_sec": round(time.time() - t0, 1),
              "device": str(device), "label_map": LABEL_MAP})
    save_json(m, out_dir / "metrics.json")
    pred = frame.iloc[res["idx"]][["path", "label"]].copy()
    pred["prob_fake"] = res["probs"]
    pred["pred"] = (res["probs"] >= thr).astype(int)
    pred["pred_name"] = pred["pred"].map(dict(enumerate(CLASS_NAMES)))
    pred.to_csv(out_dir / "predictions.csv", index=False)
    title = f"{cfg['model']['name']} tahap {ckpt['stage'].upper()} ({cfg['mode']}, {split_name})"
    M.plot_confusion(m["confusion_matrix"], out_dir / "confusion_matrix.png", title)
    M.plot_roc_pr(res["labels"], res["probs"], out_dir, title)
    M.plot_examples(pred, image_root, out_dir / "examples.png", title=title)
    auc = m["roc_auc"]
    print(f"[eval {split_name}] {title}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"ROC-AUC {auc if auc is None else round(auc, 4)} | n={m['n']}")
    return m

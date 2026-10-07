"""Membaca konfigurasi YAML: base.yaml + models/<model>.yaml + pengaturan mode."""
import copy

import yaml

from .utils import PROJECT_ROOT

MODELS = ["vgg16", "efficientnet_b0", "maxvit_t", "clip_vitl14"]
BRIEF_MODELS = ["vgg16", "efficientnet_b0", "maxvit_t"]  # model dari brief dosen
MODES = ["quick", "subset", "full", "modern_quick", "modern"]


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(model: str, mode: str, overrides: dict | None = None) -> dict:
    if model not in MODELS:
        raise ValueError(f"Model tidak dikenal: {model}. Pilihan: {MODELS}")
    if mode not in MODES:
        raise ValueError(f"Mode tidak dikenal: {mode}. Pilihan: {MODES}")
    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text())
    model_cfg = yaml.safe_load((PROJECT_ROOT / f"configs/models/{model}.yaml").read_text())
    cfg = deep_merge(base, model_cfg)

    cfg["mode"] = mode
    mode_cfg = cfg["modes"][mode]
    cfg["data"]["subset"] = {k: mode_cfg.get(k) for k in ("train_per_class", "val_per_class", "test_per_class")}
    # mode dataset modern memakai manifest, augmentasi, dan sampling sendiri
    for key in ("manifest", "augment", "balance"):
        if key in mode_cfg:
            cfg["data"][key] = mode_cfg[key]
    t = cfg["train"]
    t["epochs_a"] = mode_cfg["epochs_a"]
    t["epochs_b"] = mode_cfg["epochs_b"]
    if overrides:
        cfg = deep_merge(cfg, overrides)
        t = cfg["train"]

    # Aturan catatan: lr = 10% x default optimizer. null -> hitung dari aturan.
    rule_lr = t["default_lr"] * t["lr_factor"]
    for key in ("lr_stage_a", "lr_stage_b"):
        if t.get(key) is None:
            t[key] = rule_lr
            t[key + "_source"] = f"aturan 10%: {t['lr_factor']} x {t['default_lr']}"
        else:
            t.setdefault(key + "_source", "diatur manual")
    return cfg

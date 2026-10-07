"""Utilitas umum: path proyek, seed, device, JSON."""
import json
import os
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"real": 0, "fake": 1}  # pemetaan internal; positif = fake
CLASS_NAMES = ["real", "fake"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # juga mengatur seed MPS/CUDA


def pick_device(pref: str = "auto") -> torch.device:
    """MPS (Apple Silicon) jika tersedia, lalu CUDA, lalu CPU."""
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def load_json(path):
    return json.loads(Path(path).read_text())


def env_info() -> dict:
    import torchvision
    import sklearn

    total, _, free = shutil.disk_usage(PROJECT_ROOT)
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "ram_gb": round(ram_gb, 1) if ram_gb else None,
        "disk_free_gb": round(free / 1e9, 1),
    }


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}

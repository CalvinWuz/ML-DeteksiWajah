"""Periksa lingkungan: versi, RAM, disk, dan apakah MPS (GPU Apple) benar-benar bisa dipakai.

    python scripts/check_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from faceauth.utils import PROJECT_ROOT, env_info, pick_device, save_json  # noqa: E402


def main():
    info = env_info()
    dev = pick_device()
    info["device_selected"] = str(dev)
    try:  # uji operasi nyata, bukan hanya flag ketersediaan
        a = torch.randn(256, 256, device=dev, requires_grad=True)
        (a @ a).sum().backward()
        info["device_matmul_backward_ok"] = True
    except Exception as e:  # noqa: BLE001
        info["device_matmul_backward_ok"] = f"GAGAL: {e}"
    for k, v in info.items():
        print(f"{k:<28}{v}")
    save_json(info, PROJECT_ROOT / "logs/env.json")
    print("\nDisimpan ke logs/env.json")


if __name__ == "__main__":
    main()

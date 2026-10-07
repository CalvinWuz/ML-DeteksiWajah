"""Pemeriksaan cepat arsitektur (tanpa training sungguhan).

Untuk tiap model:
  1. ringkasan arsitektur + jumlah parameter
  2. forward batch kecil di device -> shape (B, 1) dan probabilitas
  3. tahap A: satu langkah optimizer hanya mengubah parameter head; statistik BatchNorm tetap
  4. tahap B: unfreeze sesuai kebijakan; hanya head + unit terpilih yang berubah
  5. simpan -> muat ulang checkpoint -> output identik
  6. waktu per langkah training (perkiraan kecepatan)

Memakai batch gambar asli dari manifest mode quick bila ada; jika tidak, data sintetis
(data sintetis hanya membuktikan alur kode, bukan hasil pada wajah).

    python scripts/check_models.py                    # ketiga model
    python scripts/check_models.py --models vgg16 --batch-size 8
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from faceauth.config import MODELS, load_config  # noqa: E402
from faceauth.data import build_transforms, load_manifest, make_loader  # noqa: E402
from faceauth.engine import load_checkpoint, save_checkpoint  # noqa: E402
from faceauth.models import build_model, freeze_backbone, set_train_mode, summarize, unfreeze_last  # noqa: E402
from faceauth.utils import PROJECT_ROOT, count_params, pick_device, save_json, set_seed  # noqa: E402


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def bn_buffers(model):
    return {n: b.detach().cpu().clone() for n, b in model.named_buffers() if "running" in n}


def changed(before, after):
    return {n for n in before if not torch.equal(before[n], after[n])}


def one_step(spec, x, y, device, lr, batchnorm):
    opt = torch.optim.Adam([p for p in spec.model.parameters() if p.requires_grad], lr=lr)
    set_train_mode(spec, batchnorm)
    loss = nn.BCEWithLogitsLoss()(spec.model(x).squeeze(1), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return loss.item()


def get_batch(cfg, bs):
    try:
        df, root = load_manifest(load_config(cfg["model"]["name"], "quick"))
        part = df[df["split"] == "train"].groupby("label").head(bs // 2)
        loader = make_loader(part, root, build_transforms(cfg)["eval"], bs, False, 0)
        x, y, _ = next(iter(loader))
        return x, y, f"gambar asli dari manifest quick ({len(y)} gambar, {int(y.sum())} fake)"
    except FileNotFoundError:
        g = torch.Generator().manual_seed(0)
        y = torch.tensor([0.0, 1.0] * (bs // 2))
        return torch.randn(bs, 3, 224, 224, generator=g), y, "DATA SINTETIS (manifest belum ada)"


def check(model_name, device, bs, timing_steps):
    set_seed(0)
    cfg = load_config(model_name, "quick")
    rep = {"model": model_name, "device": str(device), "batch_size": bs}
    spec = build_model(cfg, pretrained=True)
    full_params = count_params(spec.model)
    freeze_backbone(spec)
    spec.model.to(device)
    text = summarize(spec, 224, device)
    rep["summary"] = text
    rep["params_total"] = full_params["total"]
    rep["params_stage_a"] = count_params(spec.model)
    print(text)

    x, y, src = get_batch(cfg, bs)
    rep["batch_source"] = src
    x, y = x.to(device), y.to(device)
    spec.model.eval()
    with torch.no_grad():
        logits = spec.model(x)
    rep["forward_output_shape"] = list(logits.shape)
    rep["forward_prob_fake"] = [round(v, 4) for v in torch.sigmoid(logits).squeeze(1).cpu().tolist()]
    assert logits.shape == (bs, 1), logits.shape

    # ---- tahap A
    head_names = {f"{n}" for n, p in spec.model.named_parameters() if p.requires_grad}
    b0, bn0 = snapshot(spec.model), bn_buffers(spec.model)
    rep["stage_a_loss"] = one_step(spec, x, y, device, cfg["train"]["lr_stage_a"], cfg["train"]["batchnorm"])
    ch = changed(b0, snapshot(spec.model))
    bn_ch = changed(bn0, bn_buffers(spec.model))
    rep["stage_a"] = {
        "changed_params": len(ch), "head_params": len(head_names),
        "only_head_changed": ch <= head_names, "all_head_changed": ch == head_names,
        "bn_running_stats_changed": len(bn_ch),
    }
    assert ch <= head_names and len(ch) > 0, "Tahap A mengubah parameter di luar head!"
    assert not bn_ch, f"Statistik BatchNorm berubah di tahap A: {sorted(bn_ch)[:3]}"

    # ---- tahap B
    u = cfg["model"]["unfreeze"]
    ureport = unfreeze_last(spec, u["policy"], u["n"])
    rep["unfreeze"] = ureport
    rep["params_stage_b"] = count_params(spec.model)
    allowed = {n for n, p in spec.model.named_parameters() if p.requires_grad}
    backbone_allowed = allowed - head_names
    b1, bn1 = snapshot(spec.model), bn_buffers(spec.model)
    rep["stage_b_loss"] = one_step(spec, x, y, device, cfg["train"]["lr_stage_b"], cfg["train"]["batchnorm"])
    ch = changed(b1, snapshot(spec.model))
    bn_ch = changed(bn1, bn_buffers(spec.model))
    rep["stage_b"] = {
        "changed_params": len(ch), "allowed_params": len(allowed),
        "backbone_params_changed": len(ch & backbone_allowed), "backbone_params_opened": len(backbone_allowed),
        "only_allowed_changed": ch <= allowed, "frozen_backbone_unchanged": not (ch - allowed),
        "bn_running_stats_changed": len(bn_ch),
    }
    assert ch <= allowed, f"Tahap B mengubah parameter beku: {sorted(ch - allowed)[:3]}"
    assert ch & backbone_allowed, "Tahap B tidak mengubah backbone yang dibuka!"
    assert not bn_ch, "Statistik BatchNorm berubah padahal kebijakan batchnorm=eval"

    # ---- checkpoint simpan/muat ulang
    spec.model.eval()
    with torch.no_grad():
        ref = spec.model(x).cpu()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ck.pt"
        save_checkpoint(p, model_state=spec.model.state_dict(), config=cfg)
        ck = load_checkpoint(p)
        spec2 = build_model(ck["config"], pretrained=False)
        spec2.model.load_state_dict(ck["model_state"])
        spec2.model.to(device).eval()
        with torch.no_grad():
            out2 = spec2.model(x).cpu()
    rep["checkpoint_roundtrip_max_abs_diff"] = float((ref - out2).abs().max())
    assert torch.allclose(ref, out2, atol=1e-5)

    # ---- waktu per langkah tahap B
    one_step(spec, x, y, device, 1e-4, cfg["train"]["batchnorm"])  # pemanasan
    t0 = time.time()
    for _ in range(timing_steps):
        one_step(spec, x, y, device, 1e-4, cfg["train"]["batchnorm"])
    dt = (time.time() - t0) / timing_steps
    rep["stage_b_sec_per_step"] = round(dt, 4)
    rep["stage_b_img_per_sec"] = round(bs / dt, 1)
    if device.type == "mps":
        rep["mps_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    print(f"\n[OK] {model_name}: forward {rep['forward_output_shape']}, tahap A hanya head berubah, "
          f"tahap B backbone terbuka berubah ({rep['stage_b']['backbone_params_changed']}/"
          f"{rep['stage_b']['backbone_params_opened']} tensor), checkpoint identik, "
          f"{rep['stage_b_img_per_sec']} img/s (tahap B, batch {bs})\n")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--timing-steps", type=int, default=5)
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")
    out = PROJECT_ROOT / "logs/checks"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in args.models:
        try:
            results[m] = check(m, device, args.batch_size, args.timing_steps)
            (out / f"{m}_summary.txt").write_text(results[m]["summary"] + "\n")
        except Exception as e:  # noqa: BLE001 - laporkan dan lanjut ke model berikutnya
            results[m] = {"model": m, "device": str(device), "error": f"{type(e).__name__}: {e}"}
            print(f"\n[GAGAL] {m} di {device}: {type(e).__name__}: {e}\n")
    save_json(results, out / f"model_checks_{device.type}.json")
    print(f"Hasil disimpan di logs/checks/model_checks_{device.type}.json")
    if any("error" in r for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

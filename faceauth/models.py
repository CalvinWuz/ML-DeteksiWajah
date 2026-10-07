"""Tiga arsitektur dengan antarmuka seragam.

Setiap model dibungkus `ModelSpec`:
  - model    : nn.Module lengkap, output 1 logit (fake = kelas positif)
  - head     : bagian classifier yang dilatih di tahap A
  - units    : daftar (nama, modul) backbone yang dihitung oleh kebijakan unfreeze
"""
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision import models as tvm

WEIGHTS = {
    "vgg16": tvm.VGG16_Weights,
    "efficientnet_b0": tvm.EfficientNet_B0_Weights,
    "maxvit_t": tvm.MaxVit_T_Weights,
}
WEIGHTED_TYPES = (nn.Conv2d, nn.Linear)
NORM_TYPES = (nn.BatchNorm2d, nn.LayerNorm)


@dataclass
class ModelSpec:
    name: str
    model: nn.Module
    head: nn.Module
    units: list = field(default_factory=list)  # [(nama, modul)] urut dari input ke output


def get_weights(cfg):
    return getattr(WEIGHTS[cfg["model"]["name"]], cfg["model"]["weights"])


# ---------------------------------------------------------------- VGG16
def build_pre_flatten(kind: str) -> nn.Module:
    """Titik konfigurasi 'tambahan sebelum Flatten'. Jenisnya belum diketahui,
    jadi hanya 'none' yang tersedia. Tambahkan cabang baru setelah diperjelas."""
    if kind in (None, "none"):
        return nn.Identity()
    raise ValueError(
        f"pre_flatten='{kind}' belum diimplementasikan. Tambahan sebelum Flatten belum "
        "diperjelas; tambahkan layer-nya di faceauth/models.py::build_pre_flatten."
    )


class VGG16Flatten(nn.Module):
    """VGG16 standar -> (tambahan opsional) -> Flatten -> Linear -> ReLU -> Dropout -> Linear(1).
    Tidak memakai Global Average Pooling."""

    def __init__(self, weights, hidden_units: int, dropout: float, pre_flatten: str, image_size: int):
        super().__init__()
        base = tvm.vgg16(weights=weights)
        self.features = base.features  # 13 conv + ReLU + 5 MaxPool (pooling bawaan)
        self.avgpool = base.avgpool    # AdaptiveAvgPool2d(7x7) bawaan torchvision: identitas untuk input 224
        self.pre_flatten = build_pre_flatten(pre_flatten)
        self.flatten = nn.Flatten()
        with torch.no_grad():  # dimensi Flatten dihitung dari output feature extractor sebenarnya
            dummy = torch.zeros(1, 3, image_size, image_size)
            flat_dim = self.flatten(self.pre_flatten(self.avgpool(self.features(dummy)))).shape[1]
        self.flat_dim = flat_dim
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        x = self.flatten(self.pre_flatten(self.avgpool(self.features(x))))
        return self.classifier(x)


def _build_vgg16(cfg, weights):
    h = cfg["model"]["head"]
    m = VGG16Flatten(weights, h["hidden_units"], h["dropout"], h["pre_flatten"], image_size=224)
    units = [(f"features.{n}", mod) for n, mod in m.features.named_children()]
    # pre_flatten adalah bagian head (dilatih sejak tahap A) dan tidak masuk hitungan unfreeze
    head = nn.ModuleList([m.pre_flatten, m.classifier])
    return ModelSpec("vgg16", m, head, units)


# ---------------------------------------------------------------- EfficientNet-B0
def _build_efficientnet_b0(cfg, weights):
    m = tvm.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(cfg["model"]["head"]["dropout"], inplace=True), nn.Linear(in_f, 1))
    units = [(f"features.{n}", mod) for n, mod in m.features.named_children()]
    return ModelSpec("efficientnet_b0", m, m.classifier, units)


# ---------------------------------------------------------------- MaxViT-T
def _build_maxvit_t(cfg, weights):
    m = tvm.maxvit_t(weights=weights)
    last = m.classifier[-1]
    m.classifier[-1] = nn.Linear(last.in_features, 1)
    units = [("stem", m.stem)] + [(f"blocks.{n}", mod) for n, mod in m.blocks.named_children()]
    return ModelSpec("maxvit_t", m, m.classifier, units)


BUILDERS = {"vgg16": _build_vgg16, "efficientnet_b0": _build_efficientnet_b0, "maxvit_t": _build_maxvit_t}


def build_model(cfg, pretrained: bool = True) -> ModelSpec:
    """pretrained=False dipakai saat bobot akan langsung ditimpa checkpoint (tanpa unduh)."""
    weights = get_weights(cfg) if pretrained else None
    return BUILDERS[cfg["model"]["name"]](cfg, weights)


# ---------------------------------------------------------------- freeze / unfreeze
def freeze_backbone(spec: ModelSpec) -> dict:
    """Tahap A: semua parameter dibekukan kecuali head."""
    for p in spec.model.parameters():
        p.requires_grad = False
    for p in spec.head.parameters():
        p.requires_grad = True
    return {"policy": "freeze_backbone", "trainable": "head saja"}


def _weighted_layers(module):
    return [m for m in module.modules() if isinstance(m, WEIGHTED_TYPES)]


def unfreeze_last(spec: ModelSpec, policy: str, n: int) -> dict:
    """Tahap B: buka sebagian akhir backbone. Head tetap trainable.

    last_n_modules : N unit terakhir pada spec.units. Untuk VGG16 unit = modul di
                     `features` (Conv2d, ReLU, MaxPool2d dihitung masing-masing satu).
                     Untuk EfficientNet/MaxViT unit = blok/stage.
    last_n_weighted: N unit terakhir yang punya parameter (VGG16: N conv terakhir).
    """
    freeze_backbone(spec)
    if policy == "last_n_modules":
        selected = spec.units[-n:]
    elif policy == "last_n_weighted":
        selected = [u for u in spec.units if any(True for _ in u[1].parameters())][-n:]
    else:
        raise ValueError(f"Kebijakan unfreeze tidak dikenal: {policy}")
    for _, mod in selected:
        for p in mod.parameters():
            p.requires_grad = True
    weighted = [type(m).__name__ for _, mod in selected for m in _weighted_layers(mod)]
    return {
        "policy": policy,
        "n": n,
        "unit_count_total": len(spec.units),
        "selected_units": [f"{name} ({type(mod).__name__})" for name, mod in selected],
        "weighted_layers_opened": len(weighted),
        "weighted_layer_types": {t: weighted.count(t) for t in sorted(set(weighted))},
        "backbone_params_opened": sum(p.numel() for _, mod in selected for p in mod.parameters()),
    }


def set_train_mode(spec: ModelSpec, batchnorm: str = "eval") -> None:
    """model.train(), lalu:
    - unit backbone yang seluruhnya beku -> eval (dropout/stochastic depth/BN mati, seperti inferensi)
    - BatchNorm: 'eval' = semua BN eval (running mean/var tidak berubah);
                 'train_unfrozen' = BN di unit yang dibuka ikut train.
    """
    spec.model.train()
    for _, mod in spec.units:
        if not any(p.requires_grad for p in mod.parameters()):
            mod.eval()
    if batchnorm == "eval":
        for m in spec.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    elif batchnorm != "train_unfrozen":
        raise ValueError(f"batchnorm harus 'eval' atau 'train_unfrozen', bukan {batchnorm}")


def trainable_names(model: nn.Module) -> list:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def summarize(spec: ModelSpec, image_size: int = 224, device="cpu") -> str:
    """Ringkasan arsitektur: modul tingkat atas, shape output, jumlah parameter."""
    model = spec.model
    rows, hooks = [], []
    top = []
    for name, mod in model.named_children():
        if isinstance(mod, nn.ModuleList):  # ModuleList tidak dipanggil langsung; uraikan isinya
            top += [(f"{name}.{n}", m) for n, m in mod.named_children()]
        else:
            top.append((name, mod))

    def make_hook(name):
        def hook(mod, inp, out):
            shape = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
            total = sum(p.numel() for p in mod.parameters())
            train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            rows.append((name, type(mod).__name__, shape, total, train))
        return hook

    for name, mod in top:
        hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()

    lines = [f"Model: {spec.name}", f"{'modul':<14}{'tipe':<22}{'output shape':<24}{'params':>14}{'trainable':>14}"]
    for name, typ, shape, total, train in rows:
        lines.append(f"{name:<14}{typ:<22}{str(shape):<24}{total:>14,}{train:>14,}")
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"TOTAL params: {total:,} | trainable: {train:,} | beku: {total - train:,}")
    if spec.name == "vgg16":
        lines.append(f"Dimensi Flatten: {model.flat_dim:,}")
        lines.append("Unit backbone (indeks features.*):")
        for name, mod in spec.units:
            lines.append(f"  {name:<14}{type(mod).__name__:<12}{'trainable' if any(p.requires_grad for p in mod.parameters()) else '-'}")
    return "\n".join(lines)

"""Aplikasi demo untuk presentasi (berjalan lokal di browser, tanpa internet).

    streamlit run scripts/demo.py

Tab "Coba gambar"    : ambil gambar test acak atau upload sendiri, lihat prediksi model.
Tab "Hasil"          : tabel perbandingan, ROC gabungan, kurva training.
Tab "Cara kerja"     : ringkasan alur untuk dijelaskan ke audiens.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from faceauth.data import build_transforms  # noqa: E402
from faceauth.engine import load_model_from_checkpoint  # noqa: E402
from faceauth.utils import CLASS_NAMES, PROJECT_ROOT, load_json, pick_device  # noqa: E402

MODELS = {"MaxViT-T": "maxvit_t", "EfficientNet-B0": "efficientnet_b0", "VGG16": "vgg16"}
STAGES = {"B (fine-tuning)": "b", "A (transfer learning)": "a"}
RUNS = [r for r in ("full", "subset") if (PROJECT_ROOT / "outputs" / r).exists()]

st.set_page_config(page_title="Deteksi Wajah AI", page_icon="🧑", layout="wide")


@st.cache_resource
def get_model(run, model, stage):
    device = pick_device()
    spec, ck = load_model_from_checkpoint(PROJECT_ROOT / f"outputs/{run}/{model}/stage_{stage}/best.pt", device)
    return spec.model, build_transforms(ck["config"])["eval"], ck["config"]["train"]["threshold"], device


@st.cache_data
def test_images():
    info = load_json(PROJECT_ROOT / "data/manifests/manifest_info.json")
    root = PROJECT_ROOT / info["image_root"]
    df = pd.read_csv(PROJECT_ROOT / "data/manifests/full.csv")
    df = df[df["split"] == "test"][["path", "label"]]
    return str(root), df.reset_index(drop=True)


@torch.no_grad()
def predict(img, run, model, stage):
    net, tf, thr, device = get_model(run, model, stage)
    prob = torch.sigmoid(net(tf(img.convert("RGB")).unsqueeze(0).to(device))).item()
    return prob, CLASS_NAMES[int(prob >= thr)]


def show_result(prob, pred, true_label=None):
    big = "🤖 BUATAN AI (fake)" if pred == "fake" else "📷 FOTO ASLI (real)"
    st.markdown(f"## {big}")
    st.progress(prob, text=f"Probabilitas buatan AI: {prob:.1%}")
    if true_label is not None:
        ok = pred == true_label
        st.markdown(f"Label sebenarnya: **{true_label}**: " + ("✅ tebakan benar" if ok else "❌ tebakan salah"))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Pengaturan")
if not RUNS:
    st.error("Belum ada hasil training di folder outputs/. Jalankan scripts/train.py dulu.")
    st.stop()
run = st.sidebar.selectbox("Hasil training", RUNS, help="full = seluruh dataset, subset = 2.000 gambar latih")
model_label = st.sidebar.selectbox("Model", list(MODELS))
stage_label = st.sidebar.radio("Tahap", list(STAGES))
model, stage = MODELS[model_label], STAGES[stage_label]
compare_all = st.sidebar.checkbox("Bandingkan semua model", value=False)
st.sidebar.caption("Model dilatih dengan wajah asli (FFHQ) vs wajah buatan StyleGAN. "
                   "Gambar jenis lain (generator lain, foto tidak dipotong ke wajah) bisa ditebak tidak andal.")

st.title("Deteksi Wajah Asli vs Buatan AI")
tab_try, tab_res, tab_how = st.tabs(["Coba gambar", "Hasil eksperimen", "Cara kerja"])

# ---------------------------------------------------------------- tab: coba
with tab_try:
    src = st.radio("Sumber gambar", ["Gambar acak dari data test", "Upload gambar sendiri"], horizontal=True)
    img, true_label = None, None
    if src.startswith("Gambar acak"):
        root, df = test_images()
        c1, c2, c3 = st.columns(3)
        if c1.button("🎲 Ambil gambar acak", width="stretch"):
            st.session_state.pick = random.randrange(len(df))
        if c2.button("Acak wajah ASLI", width="stretch"):
            st.session_state.pick = int(df[df.label == 0].sample(1).index[0])
        if c3.button("Acak wajah AI", width="stretch"):
            st.session_state.pick = int(df[df.label == 1].sample(1).index[0])
        if "pick" in st.session_state:
            row = df.iloc[st.session_state.pick]
            img = Image.open(Path(root) / row["path"])
            true_label = CLASS_NAMES[int(row["label"])]
            hide = st.toggle("Sembunyikan label asli (untuk tebak-tebakan dengan audiens)", value=True)
    else:
        up = st.file_uploader("Pilih gambar wajah (jpg/png)", type=["jpg", "jpeg", "png", "webp"])
        if up:
            img = Image.open(up)
            hide = False

    if img is not None:
        left, right = st.columns([1, 1.3])
        left.image(img, width="stretch")
        with right:
            reveal = st.button("Tampilkan jawaban model", type="primary") if (true_label and hide) else True
            if reveal or st.session_state.get("revealed") == st.session_state.get("pick"):
                if true_label and hide:
                    st.session_state.revealed = st.session_state.get("pick")
                if compare_all:
                    rows = []
                    for ml, m in MODELS.items():
                        for sl, s in STAGES.items():
                            p, pr = predict(img, run, m, s)
                            rows.append({"model": ml, "tahap": sl, "prob buatan AI": f"{p:.1%}", "prediksi": pr,
                                         **({"benar?": "✅" if pr == true_label else "❌"} if true_label else {})})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    p, pr = predict(img, run, model, stage)
                    st.caption(f"Model: {model_label}, tahap {stage_label}, hasil training '{run}'")
                    show_result(p, pr, true_label)
            elif true_label and hide:
                st.info("Minta audiens menebak dulu: asli atau buatan AI? Lalu klik tombol di atas.")

# ---------------------------------------------------------------- tab: hasil
with tab_res:
    d = PROJECT_ROOT / "outputs" / run
    if (d / "comparison.csv").exists():
        c = pd.read_csv(d / "comparison.csv")
        c = c[c["status"] == "terukur"]
        def errors(r):  # dihitung dari confusion matrix, bukan dari accuracy yang dibulatkan
            cm = load_json(d / r["model"] / f"stage_{r['tahap'][0].lower()}" / "test" / "metrics.json")["confusion_matrix"]
            return cm[0][1] + cm[1][0]
        c["salah tebak"] = c.apply(errors, axis=1)
        st.subheader(f"Hasil pada data uji ({int(c['n_test'].max()):,} gambar)")
        st.dataframe(c[["model", "tahap", "accuracy", "roc_auc", "macro_f1", "salah tebak", "train_min"]],
                     hide_index=True, width="stretch")
    else:
        st.info(f"Jalankan: python scripts/compare.py --run-name {run}")
    if (d / "roc_comparison.png").exists():
        st.image(str(d / "roc_comparison.png"), caption="Kurva ROC semua model (makin menempel ke kiri-atas makin baik)")
    curves = d / model / "curves.png"
    if curves.exists():
        st.image(str(curves), caption=f"Kurva training {model_label}: garis merah = mulai fine-tuning")
    cm = d / model / f"stage_{stage}" / "test" / "confusion_matrix.png"
    ex = d / model / f"stage_{stage}" / "test" / "examples.png"
    cols = st.columns([1, 2])
    if cm.exists():
        cols[0].image(str(cm), caption="Confusion matrix")
    if ex.exists():
        cols[1].image(str(ex), caption="Contoh tebakan benar dan salah")

# ---------------------------------------------------------------- tab: cara kerja
with tab_how:
    st.markdown("""
**Jenis sistem:** machine learning, tepatnya *deep learning* untuk **klasifikasi gambar biner** (2 kelas: asli / buatan AI).
Model belajar sendiri dari 100.000 contoh berlabel (*supervised learning*); tidak ada aturan yang ditulis manual.

**Alur:**
1. **Data**: 140.000 wajah (70.000 foto asli dari FFHQ, 70.000 wajah buatan StyleGAN), dibagi train/validation/test.
2. **Model pretrained**: VGG16, EfficientNet-B0, dan MaxViT-T yang sudah belajar mengenali gambar umum dari ImageNet.
3. **Tahap A, transfer learning**: bagian pengenal fitur dibekukan, hanya classifier baru yang dilatih.
4. **Tahap B, fine-tuning**: bobot terbaik tahap A dibuka sebagian di bagian akhir lalu dilatih lagi.
5. **Evaluasi**: diuji pada 20.000 gambar yang tidak pernah dilihat saat training.

**Output:** probabilitas gambar buatan AI. Di atas 50% → ditebak *fake*.
""")
    sp = PROJECT_ROOT / "data/manifests/samples.png"
    if sp.exists():
        st.image(str(sp), caption="Contoh data: baris atas asli, baris bawah buatan AI")

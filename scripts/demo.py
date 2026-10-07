"""Aplikasi demo untuk presentasi (berjalan lokal di browser, tanpa internet).

    streamlit run scripts/demo.py

Tab "Coba gambar"    : ambil gambar test acak atau upload sendiri. Wajah dideteksi dan dipotong
                       otomatis (sama seperti saat training); foto berisi banyak orang dinilai per wajah.
Tab "Hasil"          : tabel perbandingan, hasil per grup uji, ROC gabungan, kurva training.
Tab "Cara kerja"     : ringkasan alur untuk dijelaskan ke audiens.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from faceauth.data import build_transforms  # noqa: E402
from faceauth.engine import load_model_from_checkpoint  # noqa: E402
from faceauth.faces import CROP_SIZE, MIN_FACE, crop_box, detect_faces  # noqa: E402
from faceauth.utils import CLASS_NAMES, PROJECT_ROOT, load_json, pick_device  # noqa: E402

MODEL_NAMES = {"maxvit_t": "MaxViT-T", "efficientnet_b0": "EfficientNet-B0", "vgg16": "VGG16",
               "clip_vitl14": "CLIP ViT-L/14"}
STAGES = {"B (fine-tuning)": "b", "A (transfer learning)": "a"}
RUN_INFO = {"modern": "wajah asli vs AI lama + modern (OpenFake, DeepFakeFace, StyleGAN)",
            "full": "140k: wajah asli FFHQ vs StyleGAN saja",
            "subset": "140k subset 2.000 gambar latih"}
RUNS = [r for r in RUN_INFO if (PROJECT_ROOT / "outputs" / r).exists()]
MAX_FACES = 12

st.set_page_config(page_title="Deteksi Wajah AI", page_icon="🧑", layout="wide")


def models_in(run):
    return [m for m in MODEL_NAMES if (PROJECT_ROOT / f"outputs/{run}/{m}/stage_a/best.pt").exists()]


@st.cache_resource
def get_model(run, model, stage):
    device = pick_device()
    spec, ck = load_model_from_checkpoint(PROJECT_ROOT / f"outputs/{run}/{model}/stage_{stage}/best.pt", device)
    return spec.model, build_transforms(ck["config"])["eval"], ck["config"]["train"]["threshold"], device


@st.cache_data
def test_images(run):
    """Gambar test dari manifest yang dipakai run tersebut."""
    name = "modern" if run == "modern" else "full"
    info_file = PROJECT_ROOT / f"data/manifests/{name}_info.json"
    info = load_json(info_file if info_file.exists() else PROJECT_ROOT / "data/manifests/manifest_info.json")
    df = pd.read_csv(PROJECT_ROOT / f"data/manifests/{name}.csv")
    cols = [c for c in ("path", "label", "group", "source") if c in df.columns]
    return str(PROJECT_ROOT / info["image_root"]), df[df["split"] == "test"][cols].reset_index(drop=True)


def find_faces(img: Image.Image):
    """[(kotak, potongan wajah 256x256 PIL)] urut dari wajah terbesar."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    out = []
    for b in detect_faces(bgr)[:MAX_FACES]:
        if max(b[2], b[3]) < MIN_FACE:
            continue
        x0, y0, side = crop_box(b, w, h)
        crop = cv2.resize(rgb[y0:y0 + side, x0:x0 + side], (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_AREA if side >= CROP_SIZE else cv2.INTER_CUBIC)
        out.append((b, Image.fromarray(crop)))
    return out


@torch.no_grad()
def predict(face, run, model, stage):
    net, tf, thr, device = get_model(run, model, stage)
    prob = torch.sigmoid(net(tf(face.convert("RGB")).unsqueeze(0).to(device))).item()
    return prob, CLASS_NAMES[int(prob >= thr)]


def annotate(img, faces, preds):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 300)
    for i, ((b, _), (p, pr)) in enumerate(zip(faces, preds), 1):
        color = (220, 40, 40) if pr == "fake" else (40, 170, 70)
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=color, width=lw)
        d.text((b[0] + 4, b[1] + 4), f"{i}: {'AI' if pr == 'fake' else 'asli'} {p:.0%}", fill=color)
    return im


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
run = st.sidebar.selectbox("Hasil training", RUNS, format_func=lambda r: f"{r}: {RUN_INFO[r]}")
avail = models_in(run)
model = st.sidebar.selectbox("Model", avail, format_func=MODEL_NAMES.get)
stage_label = st.sidebar.radio("Tahap", list(STAGES))
stage = STAGES[stage_label]
compare_all = st.sidebar.checkbox("Bandingkan semua model", value=False)
st.sidebar.caption(
    "Wajah dipotong otomatis sebelum dinilai. Tidak ada detektor yang menjamin benar untuk semua "
    "generator AI, terutama generator yang tidak ada di data latih.")

st.title("Deteksi Wajah Asli vs Buatan AI")
tab_try, tab_res, tab_how = st.tabs(["Coba gambar", "Hasil eksperimen", "Cara kerja"])

# ---------------------------------------------------------------- tab: coba
with tab_try:
    src = st.radio("Sumber gambar", ["Gambar acak dari data test", "Upload gambar sendiri"], horizontal=True)
    img, true_label, hide, meta = None, None, False, ""
    if src.startswith("Gambar acak"):
        root, df = test_images(run)
        c1, c2, c3 = st.columns(3)
        if c1.button("🎲 Ambil gambar acak", width="stretch"):
            st.session_state.pick = random.randrange(len(df))
        if c2.button("Acak wajah ASLI", width="stretch"):
            st.session_state.pick = int(df[df.label == 0].sample(1).index[0])
        if c3.button("Acak wajah AI", width="stretch"):
            st.session_state.pick = int(df[df.label == 1].sample(1).index[0])
        if "pick" in st.session_state and st.session_state.pick < len(df):
            row = df.iloc[st.session_state.pick]
            img = Image.open(Path(root) / row["path"])
            true_label = CLASS_NAMES[int(row["label"])]
            if "source" in row:
                meta = f"Sumber: {row['source']} (grup uji: {row['group']})"
            hide = st.toggle("Sembunyikan label asli (untuk tebak-tebakan dengan audiens)", value=True)
    else:
        up = st.file_uploader("Pilih gambar (jpg/png/webp). Boleh berisi lebih dari satu wajah.",
                              type=["jpg", "jpeg", "png", "webp"])
        if up:
            img = Image.open(up)

    if img is not None:
        faces = find_faces(img)
        if not faces:
            st.warning("Tidak ada wajah terdeteksi. Gambar dinilai utuh; hasilnya kurang andal.")
            faces = [((0, 0, img.width, img.height, 0), img.convert("RGB").resize((CROP_SIZE, CROP_SIZE)))]
        left, right = st.columns([1.2, 1])
        reveal = True
        if true_label and hide:
            reveal = right.button("Tampilkan jawaban model", type="primary") or \
                st.session_state.get("revealed") == st.session_state.get("pick")
            if reveal:
                st.session_state.revealed = st.session_state.get("pick")
        if not reveal:
            left.image(img, width="stretch")
            right.info("Minta audiens menebak dulu: asli atau buatan AI? Lalu klik tombol di atas.")
        else:
            preds = [predict(f, run, model, stage) for _, f in faces]
            left.image(annotate(img, faces, preds), width="stretch",
                       caption=f"{len(faces)} wajah terdeteksi" + (f" (maks {MAX_FACES})" if len(faces) >= MAX_FACES else ""))
            with right:
                if meta and not hide:
                    st.caption(meta)
                st.caption(f"Model: {MODEL_NAMES[model]}, tahap {stage_label}, hasil training '{run}'")
                if compare_all:
                    st.caption("Perbandingan untuk wajah terbesar (nomor 1).")
                    rows = []
                    for m in avail:
                        for sl, s in STAGES.items():
                            p, pr = predict(faces[0][1], run, m, s)
                            rows.append({"model": MODEL_NAMES[m], "tahap": sl, "prob buatan AI": f"{p:.1%}",
                                         "prediksi": pr,
                                         **({"benar?": "✅" if pr == true_label else "❌"} if true_label else {})})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                elif len(faces) == 1:
                    show_result(*preds[0], true_label)
                    if meta and hide:
                        st.caption(meta)
                else:
                    n_ai = sum(pr == "fake" for _, pr in preds)
                    st.markdown(f"## {n_ai} dari {len(faces)} wajah ditebak buatan AI")
                    st.dataframe(pd.DataFrame([{"wajah": i, "prob buatan AI": f"{p:.1%}",
                                                "prediksi": "AI" if pr == "fake" else "asli"}
                                               for i, (p, pr) in enumerate(preds, 1)]),
                                 hide_index=True, width="stretch")
                    st.image([f for _, f in faces], width=96, caption=[str(i) for i in range(1, len(faces) + 1)])

# ---------------------------------------------------------------- tab: hasil
with tab_res:
    d = PROJECT_ROOT / "outputs" / run
    if (d / "comparison.csv").exists():
        c = pd.read_csv(d / "comparison.csv")
        c = c[c["status"] == "terukur"].copy()

        def errors(r):  # dihitung dari confusion matrix, bukan dari accuracy yang dibulatkan
            cm = load_json(d / r["model"] / f"stage_{r['tahap'][0].lower()}" / "test" / "metrics.json")["confusion_matrix"]
            return cm[0][1] + cm[1][0]
        c["salah tebak"] = c.apply(errors, axis=1)
        st.subheader(f"Hasil pada data uji ({int(c['n_test'].max()):,} gambar)")
        show_cols = ["model", "tahap", "accuracy", "roc_auc", "macro_f1", "salah tebak", "train_min"]
        show_cols += [x for x in c.columns if x.startswith("deteksi_")]
        st.dataframe(c[[x for x in show_cols if x in c.columns]], hide_index=True, width="stretch")
    else:
        st.info(f"Jalankan: python scripts/compare.py --run-name {run}")
    ps = d / model / f"stage_{stage}" / "test" / "per_source.csv"
    if ps.exists():
        st.subheader(f"Per generator / sumber: {MODEL_NAMES[model]} tahap {stage.upper()}")
        st.caption("Kolom 'benar' = proporsi ditebak benar. Untuk fake: tingkat deteksi; untuk real: tidak salah tuduh.")
        st.dataframe(pd.read_csv(ps), hide_index=True, width="stretch")
    if (d / "roc_comparison.png").exists():
        st.image(str(d / "roc_comparison.png"), caption="Kurva ROC semua model (makin menempel ke kiri-atas makin baik)")
    curves = d / model / "curves.png"
    if curves.exists():
        st.image(str(curves), caption=f"Kurva training {MODEL_NAMES[model]}: garis merah = mulai fine-tuning")
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
Model belajar sendiri dari contoh berlabel (*supervised learning*); tidak ada aturan yang ditulis manual.

**Alur:**
1. **Data**: wajah asli dan wajah buatan AI. Versi awal: 140.000 wajah (FFHQ vs StyleGAN).
   Versi modern: ditambah foto asli Pexels/LAION/IMDB-WIKI dan wajah dari puluhan generator modern
   (ChatGPT/gpt-image, Gemini/Imagen, Midjourney, Flux, Stable Diffusion, dll).
2. **Potong wajah**: setiap gambar dipotong ke wajahnya dengan detektor wajah yang sama.
3. **Model pretrained**: VGG16, EfficientNet-B0, MaxViT-T (ImageNet), dan CLIP ViT-L/14 (perluasan).
4. **Tahap A, transfer learning**: bagian pengenal fitur dibekukan, hanya classifier baru yang dilatih.
5. **Tahap B, fine-tuning**: bobot terbaik tahap A dibuka sebagian di bagian akhir lalu dilatih lagi.
6. **Evaluasi**: diuji pada gambar yang tidak pernah dilihat, termasuk generator yang tidak ada di data latih.

**Output:** probabilitas gambar buatan AI per wajah. Di atas 50% → ditebak *fake*.
""")
    sp = PROJECT_ROOT / "data/manifests/samples.png"
    if sp.exists():
        st.image(str(sp), caption="Contoh data 140k: baris atas asli, baris bawah buatan AI")

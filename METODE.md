# Catatan metode

Dokumen ini memisahkan **ketentuan dari catatan awal**, **tafsiran kerja**, dan **pilihan implementasi**. Hal yang ditandai *belum dikonfirmasi* perlu ditanyakan ke dosen/pembimbing sebelum hasil final ditulis.

## 1. Tugas

Klasifikasi biner gambar wajah: **real** (foto wajah manusia) vs **fake** (wajah buatan AI). Model mengeluarkan satu logit; `sigmoid(logit)` = probabilitas fake. Pengenalan identitas dan pembuatan wajah bukan bagian proyek.

## 2. Ketentuan, tafsiran, dan pilihan

| Hal | Status | Yang dipakai | Di mana diatur |
|---|---|---|---|
| Model VGG16, EfficientNet, MaxViT | Ketentuan | ketiganya, VGG16 dikerjakan pertama | `configs/models/` |
| ConvNeXt Tiny/Small | Ketentuan | dibatalkan, tidak diimplementasikan | — |
| Transfer learning lalu fine-tuning, meneruskan bobot | Ketentuan (VGG16) | tahap A → `stage_a/best.pt` → tahap B | `scripts/train.py` |
| VGG16 memakai Flatten, bukan global pooling | Tafsiran kerja dari catatan | Flatten 25.088 fitur, tanpa GAP | `faceauth/models.py::VGG16Flatten` |
| "Ada tambahan sebelum Flatten" | **Belum diketahui** | tidak diisi (`pre_flatten: none`); titik konfigurasi tersedia | `vgg16.yaml`, `build_pre_flatten` |
| Dense layer boleh | Ketentuan | Linear(25088→256) → ReLU → Dropout(0,5) → Linear(256→1); ukuran 256 dan dropout 0,5 = **usulan** | `vgg16.yaml` |
| "Pre train: 10 gpp" | **Tafsiran, belum dikonfirmasi** | 10 epoch tahap A | `base.yaml: modes.*.epochs_a` |
| Epoch fine-tuning | **Belum diberikan** | 10 epoch (usulan) | `base.yaml: modes.*.epochs_b` |
| "Unfreeze 10 layer terakhir" | Ketentuan; definisi hitungan **belum dikonfirmasi** | 10 modul terakhir `features` VGG16 (lihat §4) | `vgg16.yaml: unfreeze` |
| "Learning rate 10% dari default" | Ketentuan; tahap penerapannya **belum jelas** | Adam, default 1e-3 → 1e-4 di **kedua** tahap | `base.yaml: train` |
| lr 1e-5 untuk fine-tuning | Hanya contoh yang pernah dibahas, **belum disepakati** | tidak dipakai; bisa dicoba dengan `--lr-b 1e-5` | CLI |
| Framework | Pilihan implementasi | Python + PyTorch + torchvision ("setelah tf" tidak dibaca sebagai TensorFlow) | — |
| Varian EfficientNet / MaxViT | Usulan | EfficientNet-B0, MaxViT-T | `configs/models/` |
| Dua tahap untuk EfficientNet dan MaxViT | **Perluasan rancangan** (usulan agar perbandingan konsisten) | ya, dengan unit unfreeze per blok | §5 |

## 3. Data

- Dataset: [140k Real and Fake Faces (xhlulu, Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). Real = FFHQ (Flickr, dikumpulkan NVIDIA), fake = sampel StyleGAN; 256×256 px.
- Split sumber dipertahankan (train/valid/test). Hasil audit lokal ada di `data/manifests/audit.md`.
- **Label:** CSV Kaggle memakai `real = 1`. Proyek ini me-remap ke **real = 0, fake = 1** dari nama kelas (`label_str` dan nama folder), bukan dari angka sumber. Pemetaan disimpan di manifest, checkpoint, dan `metrics.json`. Angka label sumber tetap dicatat di kolom `source_label`.
- Duplikasi diperiksa dengan md5 isi file (hanya duplikat persis; gambar mirip tidak terdeteksi).
- Mode data (sampel dipilih sekali dengan seed 42 dan dipakai semua model):

| Mode | Train | Validation | Test | Epoch A/B | Fungsi |
|---|---|---|---|---|---|
| quick | 32 + 32 | 16 + 16 | 16 + 16 | 1 / 1 | membuktikan kode tersambung; **bukan hasil** |
| subset | 1.000 + 1.000 | 500 + 500 | seluruh test | 10 / 10 | latihan awal |
| full | seluruh train | seluruh valid | seluruh test | 10 / 10 | eksperimen penuh |

  Validation diperkecil pada mode subset agar tiap epoch cepat; ini pilihan implementasi.
- **Keterbatasan:** cakupannya real (FFHQ) vs StyleGAN. Hasil bagus di sini **tidak** membuktikan kemampuan mendeteksi gambar dari generator lain (mis. diffusion) atau deepfake video. Alternatif/uji tambahan: [DeepFakeFace (OpenRL)](https://huggingface.co/datasets/OpenRL/DeepFakeFace) (`wiki.zip` vs `text2img.zip`), belum diimplementasikan.

## 4. VGG16

Paper: K. Simonyan & A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014 ([arXiv:1409.1556](https://arxiv.org/abs/1409.1556)). VGG16 standar = 13 conv + 3 FC berbobot. Di sini dipakai bobot ImageNet `IMAGENET1K_V1` dari torchvision. Tiga FC asli (4096-4096-1000) **diganti** head kecil di atas sebagai adaptasi tugas dua kelas.

Alur: `gambar → preprocessing torchvision → features (13 conv + 5 max-pool) → avgpool bawaan torchvision (7×7, identitas untuk input 224) → Flatten (512×7×7 = 25.088) → Linear(256) → ReLU → Dropout(0,5) → Linear(1)`.

**Hitungan "10 layer terakhir".** `features` torchvision berisi 31 modul (indeks 0–30). Kebijakan `last_n_modules, n=10` membuka indeks 21–30:

```
21 Conv2d(512)  ← conv4_3
22 ReLU
23 MaxPool2d
24 Conv2d(512)  ← conv5_1
25 ReLU
26 Conv2d(512)  ← conv5_2
27 ReLU
28 Conv2d(512)  ← conv5_3
29 ReLU
30 MaxPool2d
```

Jadi hanya **4 layer convolution** (≈ 9,4 juta parameter) yang benar-benar dibuka. Jika pembimbing maksudkan 10 layer **berbobot**, ganti ke `policy: last_n_weighted` (membuka conv2_2 s.d. conv5_3, 10 conv). Daftar persisnya dicetak saat training dan disimpan di `stage_b/train_info.json`.

## 5. EfficientNet-B0 dan MaxViT-T

Aturan Flatten dan "10 layer" **tidak** dipaksakan ke kedua model ini.

- **EfficientNet-B0** (Tan & Le, 2019, [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)): head bawaan dipertahankan (global average pool → Dropout 0,2 → Linear), output diganti 1 logit. Preprocessing bawaan bobot: resize 256 bicubic → crop 224. Tahap B membuka `features[6]`, `features[7]`, `features[8]` (stage MBConv 6 dan 7 + conv 1×1 akhir).
- **MaxViT-T** (Tu dkk., 2022, [arXiv:2204.01697](https://arxiv.org/abs/2204.01697)): classifier bawaan dipertahankan (pool → LayerNorm → Linear → Tanh → Linear), Linear terakhir diganti 1 logit. Input harus 224×224; preprocessing bawaan: resize 224 bicubic → crop 224. Tahap B membuka `blocks.3` (stage terakhir).
- **BatchNorm:** membekukan bobot tidak menghentikan pembaruan running mean/var di PyTorch bila modul dalam mode train. Karena itu (`batchnorm: eval`) **semua BatchNorm selalu mode eval**, dan unit backbone yang seluruhnya beku juga dijalankan mode eval (dropout/stochastic depth mati). Alasan: batch kecil (8) membuat statistik BN berisik. `scripts/check_models.py` memverifikasi statistik BN tidak berubah.

## 6. Training

- Loss `BCEWithLogitsLoss`, optimizer Adam, lr 1e-4 di kedua tahap (aturan 10%), tanpa weight decay, tanpa scheduler (lr aktual tetap dicatat per epoch di `history.csv`), tanpa early stopping (bisa diaktifkan di config).
- **Checkpoint terbaik = val_loss terendah.** Kriteria sama untuk semua model dan tahap.
- Tahap B: model dibangun tanpa bobot ImageNet lalu diisi `stage_a/best.pt` (backbone + head), lalu optimizer **dibangun ulang** berisi parameter yang trainable di tahap B.
- Augmentasi hanya di train: resize sesuai bobot → RandomCrop → RandomHorizontalFlip. Validation/test memakai transform deterministik bawaan bobot torchvision. Karena resize bawaan MaxViT sudah 224, RandomCrop pada MaxViT tidak berefek (augmentasinya hanya flip). Ini perbedaan preprocessing antar-arsitektur yang disengaja agar sesuai bobot masing-masing.
- Resume: `last.pt` disimpan tiap epoch (model, optimizer, history, RNG). Urutan acak data per epoch ditentukan `seed + epoch`.

## 7. Evaluasi

- Test set hanya dipakai **setelah** training, untuk `best.pt` tahap A dan tahap B. Test tidak dipakai memilih checkpoint, hyperparameter, atau threshold.
- Threshold tetap 0,5 (tidak dituning).
- Metrik: accuracy; precision/recall/F1 per kelas; macro-F1; confusion matrix; ROC + ROC-AUC dari `prob_fake`; precision-recall + AP (pelengkap).
- Perbandingan: `scripts/compare.py` membuat tabel 3 model × 2 tahap; yang belum dijalankan ditulis *belum tersedia*.

## 8. Rujukan

- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946.
- Tu, Z., dkk. (2022). MaxViT: Multi-Axis Vision Transformer. arXiv:2204.01697.
- Dokumentasi: [torchvision VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html), [MaxViT-T](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html), [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), [ROC-AUC scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [Transfer learning & fine-tuning (Keras guide, konsep)](https://keras.io/guides/transfer_learning/).

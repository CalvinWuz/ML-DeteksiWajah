# Log validasi

Tanggal: 6 Oktober 2026. Perangkat: MacBook M4 Pro, RAM 24 GB, macOS 27.0 arm64.
Lingkungan teruji: Python 3.11.14, torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1 (lengkap: `requirements-lock.txt`, `logs/env.json`).
Device yang dipakai di semua pemeriksaan di bawah: **MPS** (GPU Apple). CPU tidak diuji.

Tiga jenis pemeriksaan dibedakan dengan tegas:

| Jenis | Data | Arti hasilnya |
|---|---|---|
| Pemeriksaan cepat (sintetis) | gambar buatan skrip, bukan wajah | hanya membuktikan alur kode |
| Pemeriksaan cepat (`--mode quick`) | 64/32/32 gambar wajah, 1+1 epoch | alur tersambung pada data asli; angka **tidak bermakna** |
| Latihan subset (`--mode subset`) | 2.000 train / 1.000 val / 20.000 test | hasil awal yang terukur, belum hasil final |
| Eksperimen penuh (`--mode full`) | 100.000 / 20.000 / 20.000 | ketiga model selesai (7 Okt 2026) |

## 1. Lingkungan

`python scripts/check_env.py`: MPS tersedia, dan uji matmul + backward di MPS **berhasil**.

## 2. Dataset (`scripts/prepare_data.py --download`)

- Sumber: Kaggle `xhlulu/140k-real-and-fake-faces`, zip 3,75 GB, diunduh lewat Kaggle API dengan token yang sudah ada di laptop.
- 140.000 gambar terbaca dan didecode penuh: **0 rusak**, semuanya 256×256.
- Split: train 50.000 real + 50.000 fake, valid 10.000 + 10.000, test 10.000 + 10.000.
- Label sumber CSV: `real → 1`, `fake → 0`. Proyek me-remap ke **real = 0, fake = 1**. Label CSV cocok dengan nama folder (0 selisih), split CSV cocok dengan folder (0 selisih).
- Duplikasi md5 (persis): **0**, termasuk lintas split. Gambar yang mirip tetapi tidak identik tidak diperiksa.
- Bukti visual: `data/manifests/samples.png` dan `distribution.png`.

## 3. Pemeriksaan arsitektur (`scripts/check_models.py`, batch 8 gambar wajah asli: 4 real, 4 fake)

| Model | Param total | Trainable A | Trainable B | Tahap A: hanya head berubah | Tahap B: hanya head + unit terbuka berubah | BN running stats berubah | Checkpoint muat ulang | img/s tahap B (latih) | Memori MPS |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 + Flatten | 21.137.729 | 6.423.041 | 15.862.273 | ya (4/4 tensor) | ya (8/8 tensor backbone) | 0 (tidak ada BN) | identik | 137,7 | 1,18 GB |
| EfficientNet-B0 | 4.008.829 | 1.281 | 3.157.021 | ya (2/2) | ya (68/68) | 0 | identik | 183,1 | 1,29 GB |
| MaxViT-T | 30.408.137 | 264.193 | 17.794.881 | ya (6/6) | ya (82/82) | 0 | identik | 85,5 | 1,53 GB |

- VGG16: dimensi Flatten = 25.088 (512×7×7), output `(8, 1)`. Unfreeze `last_n_modules n=10` = `features.21–30` = 4 Conv2d + 4 ReLU + 2 MaxPool, berisi 9.439.232 parameter.
- Forward dan backward ketiga model **berhasil di MPS** tanpa fallback ke CPU, termasuk MaxViT.
- Sebelum dataset tersedia, pemeriksaan yang sama lolos juga dengan data sintetis (`logs/checks_synthetic.log`).

## 4. Alur training, resume, dan skrip lain (salinan proyek + data sintetis, di folder sementara)

- `train.py` A→B→evaluasi test berjalan untuk ketiga model.
- Tahap B dibangun tanpa bobot ImageNet lalu diisi `stage_a/best.pt`. `train_info.json` mencatat `init_from: stage_a/best.pt (epoch …)`.
- **Resume:** run 1 epoch lalu `--resume --epochs-a 3` melanjutkan dari epoch 2. `history.csv` berisi epoch 1–3 tanpa duplikasi. Menjalankan tanpa `--resume` ke folder yang sudah berisi hasil **ditolak**. Tahap yang sudah selesai dilewati saat resume.
- Determinisme: dua run terpisah menghasilkan train loss epoch 1 yang identik (0,23597…).
- `evaluate.py`, `predict.py`, dan `compare.py` berjalan. Notebook `00_pengantar.ipynb` dieksekusi penuh tanpa error (nbconvert), dan tidak memicu training.

## 5. Pemeriksaan cepat pada data wajah (`--mode quick`, VGG16)

Alur A→B→test berjalan pada 64 gambar train. Angkanya (acc test 0,72 → 0,63 pada 32 gambar) **tidak bermakna** karena data terlalu kecil.

## 6. Latihan subset (`--mode subset`): hasil terukur

Train 1.000 + 1.000 dan val 500 + 500 (seed 42), test penuh 20.000. Batch 8, Adam lr 1e-4 di kedua tahap, 10 + 10 epoch, checkpoint = val_loss minimum, threshold 0,5, device MPS. Tabel dari `outputs/subset/comparison.md`:

| Model | Tahap | Accuracy | ROC-AUC | AP | Macro-F1 | Epoch terbaik | Durasi latih |
|---|---|---|---|---|---|---|---|
| VGG16 | A | 0,8121 | 0,8905 | 0,8823 | 0,8120 | 10/10 | 2,6 mnt |
| VGG16 | B | 0,8708 | 0,9453 | 0,9426 | 0,8704 | 3/10 | 3,3 mnt |
| EfficientNet-B0 | A | 0,7301 | 0,8082 | 0,8030 | 0,7299 | 10/10 | 1,3 mnt |
| EfficientNet-B0 | B | 0,8859 | 0,9603 | 0,9601 | 0,8857 | 7/10 | 2,2 mnt |
| MaxViT-T | A | 0,8109 | 0,8909 | 0,8868 | 0,8109 | 10/10 | 4,3 mnt |
| MaxViT-T | B | 0,9315 | 0,9832 | 0,9833 | 0,9315 | 4/10 | 5,5 mnt |

Pengamatan (bukan kesimpulan final):

- Pada subset ini, tahap B lebih baik dari tahap A untuk ketiga model. Hasil ini belum tentu sama di dataset penuh atau dengan seed lain, dan hanya diukur sekali (satu seed).
- Tahap A: val_loss masih turun sampai epoch 10 untuk ketiga model, jadi 10 epoch kemungkinan **belum konvergen**.
- Tahap B: train acc mendekati 99%, sementara val_loss naik-turun setelah epoch terbaik. Ini tanda overfitting pada 2.000 gambar train. VGG16 juga menunjukkan train loss naik di epoch pertama tahap B (0,21 → 0,41). lr fine-tuning 1e-4 mungkin terlalu besar, tetapi dugaan ini **belum diuji** (opsi: `--lr-b 1e-5`).
- Validation hanya 1.000 gambar, jadi pemilihan checkpoint cukup berisik.

## 7. Eksperimen penuh (`--mode full`): VGG16

Dijalankan pengguna di Terminal, selesai 7 Oktober 2026 pukul 02:52. Pengaturan sama dengan subset, kecuali datanya penuh (train 100.000, val 20.000, test 20.000). Device MPS, batch 8.

| Tahap | Accuracy | ROC-AUC | AP | Macro-F1 | Confusion matrix [real, fake] | Epoch terbaik | Durasi latih |
|---|---|---|---|---|---|---|---|
| A | 0,9389 | 0,9843 | 0,9843 | 0,9389 | [[9262, 738], [484, 9516]] | 10/10 | 2,06 jam |
| B | 0,9927 | 0,9998 | 0,9998 | 0,9927 | [[9884, 116], [30, 9970]] | 10/10 | 2,42 jam |

Pengamatan:

- Epoch terbaik tahap A dan B sama-sama epoch **terakhir** (10/10). val_loss masih turun, jadi hasilnya mungkin belum konvergen.
- Tidak terlihat overfitting seperti di subset: val loss dan train loss bergerak berdekatan.
- Kesalahan tahap B lebih banyak berupa real yang ditebak fake (116) daripada fake yang ditebak real (30).
- Waktu per epoch 618–1.284 detik; satu epoch lambat (B4) kemungkinan karena laptop sedang dipakai aplikasi lain. Ini dugaan, tidak diukur.
- Hasil berlaku untuk asumsi yang masih menunggu konfirmasi dosen dan hanya untuk real (FFHQ) vs StyleGAN.

## 8. Eksperimen penuh: EfficientNet-B0

Dijalankan pengguna, selesai 7 Oktober 2026 pukul 05:58. Pengaturan sama dengan VGG16 full.

| Tahap | Accuracy | ROC-AUC | AP | Macro-F1 | Confusion matrix [real, fake] | Epoch terbaik | Durasi latih |
|---|---|---|---|---|---|---|---|
| A | 0,8377 | 0,9145 | 0,9124 | 0,8377 | [[8306, 1694], [1552, 8448]] | 10/10 | 0,86 jam |
| B | 0,9948 | 0,9998 | 0,9998 | 0,9947 | [[9935, 65], [40, 9960]] | 8/10 | 1,63 jam |

Pengamatan:

- Tahap A EfficientNet paling lemah di antara model yang sudah selesai. Hanya 1.281 parameter yang dilatih (satu Linear di atas fitur yang sudah di-pool), dan train acc (0,81) sedikit di bawah val acc (0,84). Kemungkinan underfitting; dugaan ini belum diuji.
- Setelah fine-tuning, kesalahan turun dari 3.246 menjadi 105 dari 20.000 gambar.

## 9. Eksperimen penuh: MaxViT-T dan ringkasan

MaxViT-T dijalankan pengguna, selesai 7 Oktober 2026 pukul 18:35.

| Tahap | Accuracy | ROC-AUC | AP | Macro-F1 | Confusion matrix [real, fake] | Epoch terbaik | Durasi latih |
|---|---|---|---|---|---|---|---|
| A | 0,9404 | 0,9858 | 0,9856 | 0,9404 | [[9320, 680], [512, 9488]] | 10/10 | 2,68 jam |
| B | 0,9969 | 0,9999 | 0,9999 | 0,9968 | [[9978, 22], [41, 9959]] | 10/10 | 3,68 jam |

Ringkasan run `full` (test 20.000 gambar, satu seed, threshold 0,5). Tabel lengkap: `outputs/full/comparison.md`, grafik ROC gabungan: `outputs/full/roc_comparison.png`.

| Model | Acc A | Acc B | AUC A | AUC B | Salah tebak A → B |
|---|---|---|---|---|---|
| VGG16 | 0,9389 | 0,9927 | 0,9843 | 0,9998 | 1.222 → 146 |
| EfficientNet-B0 | 0,8377 | 0,9948 | 0,9145 | 0,9998 | 3.246 → 105 |
| MaxViT-T | 0,9404 | 0,9969 | 0,9858 | 0,9999 | 1.192 → 63 |

Pengamatan:

- Pada run ini, fine-tuning menurunkan kesalahan di ketiga model. Urutan tahap B: MaxViT-T > EfficientNet-B0 > VGG16.
- Selisih antar-model di tahap B kecil (63–146 kesalahan dari 20.000), dan hanya diukur dengan satu seed. Belum ada uji signifikansi atau pengulangan seed, jadi urutan ini belum bisa disebut pasti.
- ROC-AUC tahap B semuanya ≥ 0,9998, mendekati batas atas. Untuk membedakan model, accuracy dan jumlah kesalahan lebih informatif.
- Epoch terbaik tahap A di ketiga model = 10/10, jadi tahap A kemungkinan belum konvergen. Ini relevan karena tahap B mulai dari checkpoint tersebut.

## 10. Perluasan: generator modern (7–8 Oktober 2026)

Pemicu: pengguna mengunggah kolase 10 wajah buatan aplikasi AI modern ke demo. Semua model 140k menebak "asli" (prob AI ≈ 0%). Pengujian ulang dengan memotong tiap wajah (dari screenshot, ±118 px) memberi hasil sama untuk model tahap B. Kesimpulan: kegagalan generalisasi ke generator baru, bukan masalah kolase.

Sudah diverifikasi:

- Transform evaluasi baru identik dengan preset torchvision (selisih maksimum 0,0 pada 8 gambar × 3 model), jadi hasil 140k tetap valid.
- Detektor wajah YuNet: 10/10 wajah di kolase terdeteksi; wajah FFHQ terdeteksi.
- Pembacaan parquet per halaman (`faceauth/parquet_pages.py`): header halaman terbaca, gambar terdekode dan cocok dengan baris metadata. Satu shard test OpenFake (7.031 baris) menghasilkan 944 wajah tanpa mengunduh file 5 GB utuh.
- CLIP ViT-L/14 di MPS (`check_models.py`): forward/backward OK, tahap A hanya head berubah (769 parameter), tahap B hanya 2 layer terakhir + post-LN + proyeksi berubah (25,98 juta parameter), checkpoint identik, ±25 gambar/detik saat fine-tuning, memori MPS 4,4 GB pada batch 8.
- Saringan CLIP: yang terbuang terutama anime, ilustrasi, hewan, lukisan, teks. Tingkat buang data 140k/DFF ±1% untuk real dan fake. Foto bayi asli sempat ikut terbuang, jadi prompt "foto bayi/anak" ditambahkan.
- Alur `train.py --mode modern_quick` (96 gambar latih) berjalan: sampler seimbang, augmentasi robust, evaluasi per grup dan per generator. Angkanya tidak bermakna.

Sedang berjalan (pipeline semalam, `logs/modern_pipeline.log`): pengambilan data OpenFake → `prepare_modern.py` → training 4 model mode `modern` (5+5 epoch) → `compare.py`.

## 11. Belum terverifikasi

- Pengulangan dengan seed lain dan uji signifikansi perbedaan antar-model.
- Hasil dengan seed lain, batch lebih besar, lr 1e-5, atau kebijakan `last_n_weighted`.
- Eksekusi di CPU (`--device cpu`).
- Ketahanan pada gambar di luar dataset (generator lain, foto tanpa crop wajah). Cakupan dataset hanya FFHQ vs StyleGAN.
- Hasil training mode `modern` (belum selesai) dan grup uji `lapangan` (menunggu gambar buatan pengguna).
- Interpretasi "10 gpp", "10 layer terakhir", tahap penerapan lr 10%, dan "tambahan sebelum Flatten" **masih perlu dikonfirmasi** ke pembimbing (lihat METODE.md).

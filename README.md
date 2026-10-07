# Klasifikasi wajah asli vs buatan AI

Model menerima gambar wajah dan menebak **real** (foto asli) atau **fake** (buatan AI), beserta probabilitasnya. Tiga arsitektur dibandingkan: **VGG16**, **EfficientNet-B0**, dan **MaxViT-T**. Masing-masing dilatih dua tahap:

1. **Tahap A, transfer learning.** Bagian pengenal fitur (backbone) yang sudah belajar dari ImageNet dibekukan. Hanya classifier baru (head) yang dilatih.
2. **Tahap B, fine-tuning.** Bobot terbaik tahap A dimuat, sebagian akhir backbone dibuka, lalu dilatih lagi dengan learning rate kecil.

Keputusan, asumsi, dan batasan metode ada di [METODE.md](METODE.md). Catatan pemeriksaan yang sudah dijalankan ada di [logs/VALIDASI.md](logs/VALIDASI.md).

## Isi proyek

```
configs/base.yaml            pengaturan bersama: seed, mode, batch, learning rate
configs/models/*.yaml        pengaturan tiap model (head, kebijakan unfreeze)
faceauth/                    kode inti (data, model, training, metrik)
scripts/check_env.py         cek Python, PyTorch, GPU Apple (MPS)
scripts/prepare_data.py      unduh + audit dataset, buat manifest dan contoh gambar
scripts/check_models.py      uji cepat 3 model: forward/backward, freeze/unfreeze, checkpoint
scripts/train.py             training tahap A + B, lalu evaluasi test
scripts/evaluate.py          evaluasi ulang sebuah checkpoint
scripts/predict.py           prediksi gambar baru
scripts/compare.py           tabel perbandingan model x tahap + grafik ROC gabungan
scripts/demo.py              aplikasi demo di browser untuk presentasi (potong wajah otomatis, banyak wajah)
scripts/fetch_openfake.py    ambil wajah dari dataset OpenFake (generator modern) tanpa unduh 3,4 TB
scripts/prepare_modern.py    susun dataset modern (potong wajah, saringan CLIP, grup uji)
scripts/run_modern_pipeline.sh  seluruh alur versi modern dalam satu perintah
notebooks/00_pengantar.ipynb pengantar langkah demi langkah (tidak menjalankan training)
data/                        dataset dan manifest (tidak masuk git)
outputs/                     checkpoint dan hasil (tidak masuk git)
```

## 1. Setup (sekali saja)

Buka Terminal, lalu salin perintah ini satu per satu:

```bash
cd ~/projects/face-authenticity
```

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt` berisi versi yang sudah diuji. Daftar lengkap ada di `requirements-lock.txt`.

**Setiap kali membuka Terminal baru**, aktifkan environment dulu:

```bash
cd ~/projects/face-authenticity && source .venv/bin/activate
```

Cek lingkungan (harus muncul `device_selected mps` di Mac Apple Silicon):

```bash
python scripts/check_env.py
```

## 2. Token Kaggle

Dataset diunduh dari Kaggle, jadi perlu token API. Di laptop ini file `~/.kaggle/kaggle.json` **sudah ada**, jadi langkah ini bisa dilewati. Kalau suatu saat gagal autentikasi:

1. Login di kaggle.com, buka **Settings** (klik foto profil → Settings).
2. Di bagian **API**, klik **Create New Token**. File `kaggle.json` akan terunduh.
3. Pindahkan ke tempatnya:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## 3. Siapkan dataset

Unduh (~4 GB) lalu audit otomatis:

```bash
python scripts/prepare_data.py --download
```

Kalau dataset sudah diunduh manual dan diekstrak di folder lain:

```bash
python scripts/prepare_data.py --source-dir ~/Downloads/140k-real-and-fake-faces
```

Hasilnya di `data/manifests/`:

- `audit.md`: jumlah gambar per split/kelas, file rusak, duplikasi, pemetaan label.
- `samples.png`: contoh gambar real dan fake beserta labelnya.
- `distribution.png`: grafik jumlah per split.
- `quick.csv`, `subset.csv`, `full.csv`: daftar gambar untuk tiap mode. Semua model memakai daftar yang sama.

Setelah selesai, file zip (3,7 GB) di `data/raw/140k-real-and-fake-faces/` boleh dihapus untuk menghemat ruang, karena isinya sudah diekstrak.

Label di proyek ini selalu **real = 0, fake = 1**. CSV Kaggle aslinya memakai real = 1, dan skrip sudah me-remap otomatis.

## 4. Uji cepat model

```bash
python scripts/check_models.py
```

Untuk tiap model, skrip ini mengecek: output berbentuk benar, tahap A hanya mengubah head, tahap B hanya mengubah head + blok yang dibuka, statistik BatchNorm tidak berubah, dan checkpoint bisa disimpan lalu dimuat ulang dengan hasil identik.

## 5. Training

Ada tiga mode, dan mode harus dipilih dengan `--mode`:

| Mode | Data | Epoch A / B | Gunanya |
|---|---|---|---|
| `quick` | 64 train, 32 val, 32 test | 1 / 1 | memastikan semua langkah jalan. **Angkanya bukan hasil penelitian.** |
| `subset` | 2.000 train, 1.000 val, seluruh test (20.000) | 10 / 10 | latihan awal |
| `full` | seluruh dataset (100.000 train) | 10 / 10 | eksperimen penuh, butuh berjam-jam |

Mulai dari VGG16:

```bash
python scripts/train.py --model vgg16 --mode quick
```

```bash
python scripts/train.py --model vgg16 --mode subset
```

Lalu dua model lain:

```bash
python scripts/train.py --model efficientnet_b0 --mode subset
```

```bash
python scripts/train.py --model maxvit_t --mode subset
```

Waktu yang terukur di laptop ini (M4 Pro, batch 8): mode `subset` butuh sekitar 5–16 menit per model, termasuk evaluasi 20.000 gambar test. Perkiraan mode `full` (belum diuji): VGG16 ±4 jam, EfficientNet-B0 ±2,5 jam, MaxViT-T ±6,5 jam.

Eksperimen penuh (jalankan ketika laptop bisa ditinggal lama dan dicolok ke listrik):

```bash
python scripts/train.py --model vgg16 --mode full
```

### Kalau training terputus

Jalankan perintah yang sama dengan `--resume`. Training lanjut dari epoch terakhir yang tersimpan:

```bash
python scripts/train.py --model vgg16 --mode subset --resume
```

Tanpa `--resume`, skrip menolak menimpa hasil lama. Untuk percobaan baru, beri nama run lain:

```bash
python scripts/train.py --model vgg16 --mode subset --run-name subset-coba2
```

### Pilihan yang sering dipakai

| Pilihan | Contoh | Arti |
|---|---|---|
| `--batch-size` | `--batch-size 16` | jumlah gambar per langkah. Default 8. Turunkan jika muncul error memori |
| `--stage` | `--stage b` | jalankan hanya tahap A (`a`), hanya B (`b`), atau keduanya (`both`, default) |
| `--epochs-a`, `--epochs-b` | `--epochs-b 5` | ganti jumlah epoch |
| `--lr-b` | `--lr-b 1e-5` | ganti learning rate fine-tuning. Default 1e-4 (aturan 10%). 1e-5 hanya opsi, belum disepakati |
| `--device` | `--device cpu` | paksa CPU jika GPU Apple bermasalah |

Kebijakan "unfreeze 10 layer terakhir" diatur di `configs/models/vgg16.yaml`. Penjelasannya ada di METODE.md §4.

## 6. Membaca hasil

Hasil satu model ada di `outputs/<mode>/<model>/`:

```
config.yaml                  konfigurasi lengkap yang dipakai
curves.png                   kurva loss/accuracy per epoch; garis merah = mulai fine-tuning
stage_a/  dan  stage_b/
  summary.txt                ringkasan arsitektur dan jumlah parameter
  train_info.json            lr, batch, epoch, device, durasi, layer yang dibuka
  trainable_params.txt       daftar parameter yang dilatih di tahap itu
  history.csv                angka per epoch (termasuk lr aktual)
  best.pt / last.pt          checkpoint terbaik (val_loss terendah) / terakhir
  test/
    metrics.json             accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    predictions.csv          prob_fake dan prediksi untuk tiap gambar test
    confusion_matrix.png, roc.png, pr_curve.png, examples.png
```

Cara membaca angka utama:

- **Accuracy**: persentase tebakan yang benar.
- **Precision fake**: dari gambar yang ditebak fake, berapa persen yang memang fake.
- **Recall fake**: dari semua gambar fake, berapa persen yang tertangkap.
- **F1**: gabungan precision dan recall (makin dekat 1 makin baik).
- **ROC-AUC**: kemampuan memisahkan real dan fake di semua ambang. 0,5 = tebak acak, 1,0 = sempurna.
- **Confusion matrix**: baris = label asli, kolom = tebakan.

Tabel perbandingan semua model dan tahap:

```bash
python scripts/compare.py --run-name subset
```

Hasilnya di `outputs/subset/comparison.md`, ditambah `roc_comparison.png` (kurva ROC semua model dalam satu grafik). Kombinasi yang belum dijalankan ditulis *belum tersedia*.

## 7. Prediksi gambar baru

```bash
python scripts/predict.py --checkpoint outputs/subset/vgg16/stage_b/best.pt --images foto_wajah.jpg
```

`--images` bisa berupa satu file, beberapa file, atau folder. Model dilatih dengan wajah yang sudah dipotong rapi (real vs StyleGAN), jadi gambar yang sangat berbeda bisa membuat prediksinya tidak andal.

Evaluasi ulang checkpoint:

```bash
python scripts/evaluate.py --checkpoint outputs/subset/vgg16/stage_b/best.pt
```

## 8. Versi modern: deteksi wajah AI zaman sekarang

Model versi 140k hanya mengenal wajah StyleGAN, sehingga **gagal pada wajah dari ChatGPT, Gemini, Midjourney, dan sejenisnya**. Versi modern menambahkan data dari puluhan generator terbaru, potong wajah otomatis, augmentasi tahan-kompresi, dan model ke-4 (CLIP ViT-L/14). Penjelasan lengkap ada di METODE.md §9.

Satu perintah untuk semuanya: ambil data, susun dataset, latih 4 model, lalu buat tabel. Prosesnya bisa belasan jam. Laptop jangan ditutup dan tetap dicolok charger:

```bash
nohup caffeinate -i -s bash scripts/run_modern_pipeline.sh > logs/modern_pipeline.log 2>&1 &
```

Lihat progresnya:

```bash
tail -5 logs/modern_pipeline.log
```

Kalau terputus, jalankan perintah yang sama lagi. Bagian yang sudah selesai akan dilewati.

### Uji dengan gambar Gemini/ChatGPT buatanmu sendiri

1. Buka daftar prompt: `open data/user_generated/prompts.csv`
2. Buat gambar di aplikasi Gemini atau ChatGPT, **unduh file aslinya (jangan screenshot)**, lalu simpan ke `data/user_generated/gemini/` atau `data/user_generated/chatgpt/`, misalnya `001.png`.
3. Setelah pipeline selesai, susun ulang dataset supaya gambarmu masuk sebagai grup uji `lapangan`:
   ```bash
   python scripts/prepare_modern.py
   ```
4. Evaluasi ulang model, misalnya CLIP:
   ```bash
   python scripts/evaluate.py --checkpoint outputs/modern/clip_vitl14/stage_b/best.pt
   ```
   Hasil per generator ada di `outputs/modern/clip_vitl14/stage_b/test_eval/per_source.csv`.

## 9. Aplikasi demo (untuk presentasi)

```bash
streamlit run scripts/demo.py
```

Browser otomatis terbuka di `http://localhost:8501`. Semuanya berjalan lokal, tanpa internet dan tanpa kamera.

- **Coba gambar**: ambil wajah acak dari data test (bisa disembunyikan labelnya untuk tebak-tebakan dengan audiens), atau upload gambar sendiri. Wajah dideteksi dan dipotong otomatis; foto berisi banyak orang atau kolase dinilai per wajah. Pilih hasil training `modern` untuk model yang mengenal generator terbaru.
- **Hasil eksperimen**: tabel perbandingan, kurva ROC, kurva training, confusion matrix.
- **Cara kerja**: ringkasan alur untuk dijelaskan.

Untuk menutup aplikasi, tekan `Ctrl+C` di Terminal.

## 10. Notebook

```bash
python -m ipykernel install --user --name face-authenticity --display-name "Python (face-authenticity)"
```

Buka `notebooks/00_pengantar.ipynb` di VS Code atau Jupyter, pilih kernel **Python (face-authenticity)**. Notebook menampilkan data, label, arsitektur VGG16, dan prediksi pada batch kecil. Notebook ini tidak menjalankan training.

## 11. Masalah umum

- **`MPS backend out of memory`**: turunkan batch, misalnya `--batch-size 4`.
- **Operasi tidak didukung MPS**: jalankan dengan `--device cpu` (lebih lambat), atau tambahkan `PYTORCH_ENABLE_MPS_FALLBACK=1` di depan perintah.
- **`Manifest ... belum ada`**: jalankan langkah 3 dulu.
- **`... sudah berisi hasil`**: tambahkan `--resume` atau pakai `--run-name` lain.

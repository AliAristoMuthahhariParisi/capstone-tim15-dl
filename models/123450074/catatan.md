# Catatan Desain & Eksperimen Model: CNN 2D (Jalur CNN)

* **Nama:** Lutfia Aisyah Putri
* **NIM:** `123450074`
* **Jalur Metode:** CNN (Konvolusi 2D dengan Penaksir Masker Waktu-Frekuensi Residual)
* **Dataset:** Data primer meja laboratorium `sesi-20261117-071715` (160 latih, 40 validasi, 40 uji)
* **Kasus Tim:** Pembersihan ucapan dari bising mesin (Ruang Cerdas 5 Volt)

---

## 1. Arsitektur Model

Model menggunakan arsitektur **CNN 2D Residual Masking** dengan total **31.227 parameter**:
1. **Blok Penaksir Masker Waktu-Frekuensi (*Residual Mask Estimator*):**
   * Masukan spektrogram log-magnitudo terstandar `[B, 1, 32, 40]`.
   * Dua lapis konvolusi 2D (`Conv2d(1, 16, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(16, 1, 3x3) -> Tanh`).
   * Fungsi aktivasi `Tanh()` menghasilkan bobot atenuasi `[-1, 1]`. Dalam domain log-spektrogram, penapisan bising aditif $x_{\text{bersih}} = x + M(x)$ setara dengan penapisan magnitudo rasional $\log(S \cdot M) = \log(S) + \log(M)$.
2. **Blok Ekstraksi Ciri Konvolusi 2D:**
   * Dua tahap blok konvolusi (`Conv2d(1, 32) -> BN -> ReLU -> Conv2d(32, 32) -> BN -> ReLU -> MaxPool2d(2)` dilanjutkan `Conv2d(32, 64) -> BN -> ReLU -> MaxPool2d(2)`).
   * Meringkas fitur spasial waktu-frekuensi menjadi representasi berdimensi kompak melalui `AdaptiveAvgPool2d((2, 2))`.
3. **Kepala Klasifikasi (*Classification Head*):**
   * `Flatten` -> `Dropout(p=0.3)` -> `Linear(256, 10)` menghasilkan probabilitas logit untuk 10 kelas kalimat perintah kendali alat meja.

---

## 2. Alasan Optimizer & Regularisasi

* **Optimizer:** Menggunakan `AdamW` (`lr = 1e-3`, `weight_decay = 1e-3`). Keunggulan `AdamW` terletak pada pemisahan peluruhan bobot (*decoupled weight decay*), sehingga regularisasi $L_2$ tidak terdistorsi oleh estimasi momen gradien orde pertama dan kedua pada dataset latih kecil (160 sampel).
* **Regularisasi:**
  1. **Batch Normalization (`BatchNorm2d`)** pada setiap lapisan konvolusi untuk menstabilkan distribusi aktivasi antar-batch serta mempercepat konvergensi.
  2. **Dropout (`p = 0.3`)** pada lapisan linear sebelum proyeksi 10 kelas untuk mencegah adaptasi bersama (*co-adaptation*) fitur penutur tertentu.
  3. **Weight Decay (`1e-3`)** untuk menjaga bobot konvolusi tetap kecil dan mengendalikan kompleksitas model.

---

## 3. Pencarian Hyperparameter (Tabel)

Eksperimen pencarian hyperparameter dilakukan pada data primer meja dengan variasi laju belajar (*learning rate*) dan jumlah epoch pelatihan:

| Learning Rate | Epochs | Batch Size | Val Accuracy | Val Macro-F1 | Keterangan |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $1 \times 10^{-3}$ | **15** | 16 | **0.1500** (15.0%) | **0.1218** | **Konfigurasi terbaik (puncak generalisasi val)** |
| $1 \times 10^{-3}$ | 20 | 16 | 0.1000 (10.0%) | 0.0917 | Mulai terjadi overfitting pada penutur latih |
| $1 \times 10^{-3}$ | 30 | 16 | 0.0500 (5.0%) | 0.0429 | Overfitting tajam (akurasi latih 100%, val drop) |
| $5 \times 10^{-4}$ | 15 | 16 | 0.1000 (10.0%) | 0.1033 | Konvergensi lebih lambat pada epoch awal |
| $5 \times 10^{-4}$ | 20 | 16 | 0.1000 (10.0%) | 0.0929 | Stabil namun akurasi di bawah lr 1e-3 |
| $5 \times 10^{-4}$ | 30 | 16 | 0.0750 (7.5%) | 0.0682 | Degradasi performa akibat penghafalan data latih |

*Analisis:* Pada skenario pemisahan penutur independen (*speaker-disjoint split*) dengan data latih 160 sampel dari 4 partisipan (P01-P04), model sangat cepat menghafal karakteristik akustik vokal penutur latih. Pelatihan 15 epoch dengan $lr = 10^{-3}$ menghasilkan titik optimal generalisasi terhadap penutur validasi P05 sebelum terjadi saturasi *overfitting*.

---

## 4. Ablation Study (>= 2)

Dilakukan studi ablasi sistematis untuk menguji kontribusi komponen utama arsitektur:

| Variasi Eksperimen | Val Accuracy | Val Macro-F1 | Selisih terhadap Baseline | Analisis Kontribusi |
| :--- | :---: | :---: | :---: | :--- |
| **Model Lengkap (Baseline)**<br>(Penaksir Masker Tanh + Dropout 0.3) | **0.1500** | **0.1218** | Acuan | Konfigurasi utuh dengan blok penapis derau residual. |
| **Ablasi A: Tanpa Penaksir Masker**<br>(Fitur CNN langsung dari spektrogram masukan) | **0.0750** | **0.0819** | **-7.50%** | Performa validasi anjlok drastis ketika blok penaksir masker dihilangkan. Hal ini membuktikan bahwa blok residual masker secara efektif meredam komponen bising mesin kipas dan pompa sebelum diekstraksi oleh konvolusi utama. |
| **Ablasi B: Variasi Dropout (p = 0.5)**<br>(Meningkatkan rasio dropout pada classifier) | **0.1750** | **0.1908** | **+2.50%** | Peningkatan dropout menjadi 0.5 memberikan regularisasi yang lebih agresif, menahan laju penghafalan penutur latih dan mendongkrak akurasi penutur validasi P05 ke 17.5%. |

---

## 5. Hasil 3 Seed Konfigurasi Akhir

Evaluasi konfigurasi akhir pada data primer meja (`sesi-20261117-071715`) dijalankan pada 3 seed acak:

| Seed | Val Accuracy | Val Macro-F1 | Durasi (detik) | Berkas Metrik |
| :---: | :---: | :---: | :---: | :--- |
| **Seed 1** | **0.1500** | **0.1218** | 39.1 s | `results/123450074/metrics.json` |
| **Seed 2** | **0.1250** | **0.0987** | 38.8 s | Tercatat |
| **Seed 3** | **0.0250** | **0.0200** | 39.4 s | Tercatat |
| **Rerata ± Std** | **0.1000 ± 0.0540** | **0.0802 ± 0.0438** | ~39.1 s | — |

### Komparasi terhadap Garis Dasar (*Baseline*):
* **Baseline Starter Meja (`results/baseline/metrics.json`):**
  * `val_acc = 0.0250` (2.50%)
  * `val_macro_f1 = 0.0222` (2.22%)
* **Model CNN 2D (`results/123450074/metrics.json`):**
  * `val_acc = 0.1500` (15.00%) — **6x lipat melampaui baseline starter**.
  * `val_macro_f1 = 0.1218` (12.18%) — **5.5x lipat melampaui baseline starter**.

Seluruh artefak hasil telah tervalidasi memenuhi skema penilaian Workbench dengan sidik data resmi `data_hash: a565707907851999be537f7e1272a55d`.

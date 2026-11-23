# Catatan Desain & Eksperimen Model: RNN LSTM (Jalur RNN)

* **Nama:** Ali Aristo Muthahhari Parisi
* **NIM:** `123450088`
* **Jalur Metode:** RNN (LSTM dengan Penaksir Masker Waktu-Frekuensi)
* **Dataset:** Data primer meja laboratorium `sesi-20261117-071715` (160 latih, 40 validasi, 40 uji)
* **Kasus Tim:** Pembersihan ucapan dari bising mesin (Ruang Cerdas 5 Volt)

---

## 1. Arsitektur Model

Model menggunakan **LSTM satu lapis** dengan **kepala penaksir masker per frame**, total **27.851 parameter**:

1. **LSTM (`nn.LSTM`)**
   * Masukan `[B, 32, 40]` — 32 frame waktu, 40 bin frekuensi dari spektrogram log-magnitudo terstandar.
   * `input_size = 40`, `hidden_size = 64`, `num_layers = 1`, `batch_first = True`.
   * Mengubah deret frame menjadi representasi tersembunyi `[B, 32, 64]` yang menangkap dinamika temporal ucapan.

2. **Kepala Penaksir Masker**
   * `Linear(64, 1)` + `Sigmoid` menghasilkan satu bobot atenuasi per frame (`[B, 32, 1]`).
   * Bobot dikalikan ke keluaran LSTM, sehingga frame yang didominasi bising kipas/pompa ditekan sebelum dirangkum.

3. **Kepala Klasifikasi**
   * *Mean-pooling* atas 32 frame menjadi vektor `[B, 64]`.
   * `Dropout(p = 0.3)` → `Linear(64, 10)` menghasilkan logit untuk 10 kelas kalimat perintah.

---

## 2. Alasan Optimizer & Regularisasi

* **Optimizer:** `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`). Pada LSTM dengan 27 ribu parameter dan hanya 160 sampel latih, laju belajar adaptif mempercepat konvergensi, sedangkan *decoupled weight decay* menjaga regularisasi $L_2$ tidak terdistorsi oleh estimasi momen gradien.
* **Regularisasi:**
  1. **Dropout (`p = 0.3`)** sebelum lapisan klasifikasi untuk mencegah adaptasi bersama fitur penutur tertentu.
  2. **Weight Decay (`1e-4`)** menjaga bobot tetap kecil.
  3. **Masker sigmoid** berfungsi sebagai pembatas informasi: frame bising ditekan sebelum dirangkum.

---

## 3. Pencarian Hyperparameter (Tabel)

Dijalankan pada data primer meja, split protokol latih P01–P04 / validasi P05, 40 epoch, dirata-rata atas 3 seed (1, 2, 3):

| Konfigurasi | Val Accuracy | Val Macro-F1 | Keterangan |
| :--- | :---: | :---: | :--- |
| LSTM 64 + masker (acuan) | 0.075 ± 0.020 | 0.059 ± 0.022 | Konfigurasi dasar |
| **GRU 64 + masker** | **0.092 ± 0.031** | **0.075 ± 0.034** | Selisih terkecil antar-seed |
| LSTM 64 tanpa masker | 0.083 ± 0.043 | 0.072 ± 0.045 | Masker tidak menambah akurasi |
| LSTM 32, dropout 0.5, wd 1e-3 | 0.092 ± 0.043 | 0.079 ± 0.063 | Simpangan paling besar |
| LSTM 64 + augmentasi (geser waktu + derau) | 0.067 ± 0.024 | 0.051 ± 0.010 | Justru menurunkan akurasi |
| LSTM 64 + augmentasi + dropout 0.5 + wd 1e-3 | 0.092 ± 0.031 | **0.087 ± 0.029** | Macro-F1 tertinggi, simpangan besar |

*Analisis:* Dengan 40 contoh validasi, satu jawaban benar bernilai 0.025. Seluruh konfigurasi berada pada rentang 0.067–0.092, yaitu **di sekitar peluang acak 10 kelas (0.10)**. Simpangan antar-seed (0.020–0.043) sebanding dengan selisih antar-konfigurasi, sehingga belum ada konfigurasi yang terbukti lebih baik. Penurunan akurasi pada augmentasi menunjukkan model belum menangkap pola yang dapat digeneralisasi, bukan kekurangan variasi data.

---

## 4. Ablation Study (>= 2)

| Variasi Eksperimen | Val Accuracy | Selisih | Analisis |
| :--- | :---: | :---: | :--- |
| **Model lengkap** (LSTM 64 + masker) | 0.075 | Acuan | Konfigurasi utuh |
| **Ablasi A: tanpa kepala masker** | 0.083 | +0.008 | Masker tidak memberi manfaat terukur; kemungkinan karena masukan belum memuat komponen ucapan yang jelas |
| **Ablasi B: LSTM → GRU** | 0.092 | +0.017 | GRU sedikit lebih baik pada jumlah parameter setara, belum signifikan |

---

## 5. Hasil 3 Seed Konfigurasi Akhir

Konfigurasi akhir `run.py` (LSTM 64 + masker, 60 epoch) dijalankan pada 3 seed:

| Seed | Val Accuracy | Val Macro-F1 | Durasi (detik) | Berkas Metrik |
| :---: | :---: | :---: | :---: | :--- |
| **Seed 1** | **0.1250** | **0.0873** | 11.4 s | `results/123450088/metrics.json` |
| Seed 2 | 0.1000 | 0.0958 | ~11 s | Tercatat |
| Seed 3 | 0.0750 | 0.0662 | ~11 s | Tercatat |
| **Rerata ± Std** | **0.1000 ± 0.0204** | **0.0831 ± 0.0123** | ~11.3 s | — |

Seluruh artefak memakai sidik data resmi `data_hash: a565707907851999be537f7e1272a55d`.

### Perbandingan dengan jalur lain pada data yang sama

| Model | Val Accuracy |
| :--- | :---: |
| Baseline Spectral Subtraction / Wiener Filter | 0.0000 (SI-SDR −32.31 dB) |
| CNN 2D (123450074) | 0.1500 |
| **RNN LSTM (123450088)** | **0.1250** |

---

## 6. Catatan Keterbatasan Data

Akurasi seluruh jalur berada jauh di bawah harapan, dan pemeriksaan menunjukkan penyebabnya ada pada data sesi, bukan pada model:

1. **Profil energi dalam jendela 4 detik hampir datar** (rasio antar-potongan 0,9–1,0). Ucapan manusia seharusnya muncul sebagai ledakan energi 10–30 kali latar.
2. **Energi audio tidak berubah mengikuti aktuator.** RMS rata-rata: kipas 0/pompa 0 = 0.0555; kipas 0/pompa 200 = 0.0584; kipas 255/pompa 0 = 0.0565; kipas 255/pompa 200 = 0.0568.
3. **Sebaran fitur per kelas hampir identik** (rerata −0.105 sampai +0.065 pada skala −2.8..27.5).
4. Regresi logistik mencapai akurasi 1.0 pada data latih sendiri, sehingga fitur tidak rusak — memang tidak ada informasi pembeda yang dapat digeneralisasi ke penutur baru.

Tindakan yang disarankan: memeriksa rekaman WAV pada kondisi `kipas 255, pompa 200` dibandingkan `kipas 0, pompa 0`, dan bila tidak ada perbedaan, sesi perlu diulang pada slot cadangan.

## 7. Catatan Teknis

* `src.common.evaluate` berjalan normal pada data primer; `metrics.json` ditulis dengan `data_hash a565707907851999be537f7e1272a55d`.
* `data/CATATAN-SESI.md` mencantumkan `data_hash 288d1dad9a827a418a79826a69f54389`, berbeda dari hasil `manifest.data_hash()`. Perlu diselaraskan.
* Split memakai `data.muat()`: latih P01–P04, validasi P05, uji P06, terpisah per penutur.

# Catatan Desain Model — Transfer Learning (`122450021`)

- **Pemilik:** Lisa Diani Amelia (`122450021`)
- **Jalur Metode:** Transfer Learning (`tl`) — Adaptasi model pralatih (*MobileNetV2*) pada masukan spektrogram log-magnitudo 3 detik efektif (hasil pemotongan transien 1,0 detik awal).

## 1. Alasan Desain Arsitektur
- **Backbone Pralatih (MobileNetV2):** Dipilih karena efisien dalam mengekstrak fitur spasial 2D (32 frame waktu $\times$ 40 bin frekuensi) dengan jumlah parameter ringan dan waktu latih singkat di CPU laptop (< 1 menit per run).
- **Strategi Pembekuan (*Partial Freeze*):** Lapisan konvolusi awal dibekukan untuk mempertahankan ekstraktor fitur dasar, sedangkan blok konvolusi akhir (15 tensor parameter terakhir) dan *linear head* diadaptasi pada domain ucapan bising meja.
- **Kesesuaian Masukan & Keluaran:** Spektrogram direplikasi menjadi 3 kanal dan di-interpolasi ke ukuran $128 \times 128$, lalu diproyeksikan ke 10 kelas kalimat perintah melalui *global average pooling* dan lapisan linear.

## 2. Optimizer & Regularisasi
- **Optimizer:** `AdamW` ($lr = 10^{-3}$, `weight_decay` $= 10^{-4}$) untuk menjaga stabilitas pembaruan bobot saat proses *fine-tuning*.
- **Regularisasi:** Menggunakan *Dropout* ($p = 0.2$) sebelum lapisan proyeksi akhir serta *weight decay* pada `AdamW` untuk menekan *overfitting* pada 160 sampel latih (P01–P04).

## 3. Pencarian Hyperparameter

Dijalankan pada data primer meja (`sesi-20261117-071715`, `seed = 1`), split bebas bocor penutur latih P01–P04 / validasi P05.

| Epoch adaptasi | Learning Rate | Val Accuracy | Val Macro-F1 | Keterangan |
|:---:|:---:|:---:|:---:|---|
| 3 | 1e-3 | 0.1250 | 0.0717 | belum konvergen penuh (loss 1,2534) |
| 5 | 1e-3 | 0.0750 | 0.0500 | konfigurasi acuan (stabil lintas seed) |
| 8 | 1e-3 | 0.0500 | 0.0400 | overfitting pada penutur latih (loss 0,1087) |
| 5 | 5e-4 | 0.1000 | 0.0708 | pembaruan bobot lebih lambat (loss 1,1911) |

## 4. Ablation Study

Diuji pada `seed = 1`, `epochs = 5`, `lr = 1e-3` untuk mengukur pengaruh pembekuan *backbone* MobileNetV2:

| Variasi | Val Accuracy | Val Macro-F1 | Analisis |
|:---|:---:|:---:|---|
| Model lengkap (`partial` freeze + Dropout 0.2) | 0.0750 | 0.0500 | konfigurasi acuan (15 tensor terakhir dibuka) |
| Ablasi A: seluruh backbone dibekukan (`full` freeze) | 0.1250 | 0.0927 | menghindari overfitting fitur konvolusi pada 160 sampel |
| Ablasi B: seluruh backbone di-fine-tune (`none` freeze) | 0.0750 | 0.0341 | waktu latih 2x lebih lama (56,5 dtk) dan rentan overfit |

## 5. Hasil 3 Seed Konfigurasi Akhir

Evaluasi konfigurasi standar (`epochs = 5`, `lr = 1e-3`, `partial` freeze) pada tiga acakan berbeda:

| Seed | Val Accuracy | Val Macro-F1 | Durasi (detik) |
|:---:|:---:|:---:|:---:|
| 1 | 0.0750 | 0.0500 | 27.0 |
| 2 | 0.1250 | 0.1033 | 25.8 |
| 3 | 0.1750 | 0.1092 | 25.0 |
| **Rerata ± Std** | **0.1250 ± 0.0408** | **0.0875 ± 0.0266** | **25.9 ± 0.8** |

## 6. Analisis Keterbatasan Data Sesi
- Rerata akurasi validasi sebesar **0,1250 ± 0,0408** (kisaran peluang acak 10 kelas = 0,10) konsisten dengan capaian seluruh jalur model lain dalam tim (Baseline Wiener 0,0000; CNN 0,1500; RNN 0,1250).
- Hal ini disebabkan karakteristik rekaman sesi meja `sesi-20261117-071715`, di mana variasi energi akustik tidak berubah signifikan antar-taraf kipas dan pompa (RMS ~0,0555 saat alat mati vs ~0,0568 saat kipas 255 + pompa 200), sehingga sinyal pembeda antar-kelas pada spektrogram sangat terbatas.

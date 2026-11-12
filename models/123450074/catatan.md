# Catatan model 123450074 (Jalur CNN)

## Alasan optimizer
Menggunakan `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) karena konvergensinya lebih cepat dan stabil pada fitur spektrogram 2D dibanding SGD polos tanpa perlu tuning momentum manual, serta memisahkan peluruhan bobot (*decoupled weight decay*) agar regularisasi tidak terdistorsi oleh laju belajar adaptif.

## Regularisasi
1. **Batch Normalization (`BatchNorm2d`)** pada setiap blok konvolusi (penaksir masker dan ekstraksi ciri) untuk menstabilkan distribusi aktivasi antar-batch.
2. **Dropout (`p=0.3`)** sebelum lapisan klasifikasi akhir untuk mencegah *co-adaptation* fitur pada ukuran data latih yang kecil.
3. **Weight Decay (`1e-4`)** melalui optimizer `AdamW`.

## Pencarian hyperparameter (tabel)
*(Akan diisi pada eksperimen data primer meja uji)*

## Ablation (>= 2)
*(Akan diisi pada eksperimen data primer meja uji — mis. tanpa blok penaksir masker vs dengan penaksir masker, variasi ukuran kernel / kedalaman kanal)*

## Hasil 3 seed konfigurasi akhir
*(Uji kerangka pada data simulasi: `seed=1`, 30 epoch, tersimpan di `results/123450074/metrics_sintetis.json` sebagai verifikasi pipeline saja — bukan hasil capaian).*

# Catatan model 123450088 (Jalur RNN)

## Alasan optimizer
`AdamW` (`lr=1e-3`, `weight_decay=1e-4`). Pada LSTM dengan 27 ribu parameter dan 160 contoh latih, momentum adaptif mempercepat konvergensi dibanding SGD, sementara *decoupled weight decay* menjaga regularisasi tidak terdistorsi oleh laju belajar adaptif.

## Arsitektur
- **LSTM** (`input=40` bin frekuensi, `hidden=64`, 1 lapis, `batch_first`) membaca 32 frame sebagai deret waktu.
- **Kepala masker time-frequency**: satu bobot sigmoid per frame, dikalikan ke keluaran LSTM — meniru peran penaksir masker spektrogram seperti pada jalur CNN.
- **Pooling rata-rata** atas 32 frame, lalu Dropout 0.3 dan satu lapis Linear 10 kelas.

## Regularisasi
1. **Dropout (`p=0.3`)** sebelum lapisan klasifikasi, karena data latih kecil (160 contoh).
2. **Weight Decay (`1e-4`)** lewat `AdamW`.
3. **Masker sigmoid** berfungsi sebagai pembatas informasi: frame yang bising ditekan sebelum dirangkum.

## Pencarian hyperparameter (tabel)
*(Akan diisi pada eksperimen data primer meja uji — kandidat: hidden 32/64/128, dropout 0.2/0.3/0.5, lr 1e-3/5e-4)*

## Ablation (>= 2)
*(Akan diisi setelah data primer: (a) tanpa kepala masker vs dengan kepala masker, (b) LSTM vs GRU pada jumlah parameter setara)*

## Hasil pada data simulasi
Uji kerangka `seed=1`, 12 epoch, data jadwal gladi: `val_acc 0.30`, `macro_f1 0.235`, tersimpan di `results/123450088/metrics_sintetis.json`. Angka ini **verifikasi pipeline**, bukan capaian — data masih sintetis.

## Catatan teknis
- `src.common.evaluate` memerlukan `data/spektrogram_rekaman.npz` (tercantum di `data/MANIFEST.json`) untuk menghitung `data_hash`. Berkas itu tidak ada di mesin pengembangan, sehingga `run.py` menulis metrik cadangan ke `metrics_sintetis.json` dan mencetak penyebabnya.
- Split memakai `data.muat()`: latih P01–P04, validasi P05, uji P06 — terpisah per penutur, sesuai protokol.

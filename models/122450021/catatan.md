# Catatan Desain Model — Transfer Learning (`122450021`)

- **Pemilik:** Lisa Diani Amelia (`122450021`)
- **Jalur Metode:** Transfer Learning (TL) — Adaptasi model pralatih (*MobileNetV2*) pada masukan spektrogram magnitudo 4 detik.

## 1. Alasan Desain Arsitektur
- **Backbone Pralatih (MobileNetV2):** Dipilih karena efisien dalam mengekstrak pola tekstur spasial 2D (frekuensi $\times$ waktu) pada spektrogram magnitudo 4 detik dengan jumlah parameter yang ringan untuk ukuran dataset eksperimen tim.
- **Strategi Pembekuan (*Partial Freeze*):** Lapisan konvolusi awal dibekukan untuk mempertahankan ekstraktor fitur spasial dasar, sementara blok konvolusi akhir dan *linear head* diadaptasi guna menangkap pola masker bising dan struktur sinyal target.
- **Kesesuaian Masukan & Keluaran:** Input spektrogram 4 detik dikonversi ke representasi 3 kanal dan di-interpolasi ke resolusi spasial standar backbone, sementara dimensi keluaran menyesuaikan bentuk target (`y_val`/`y_test`) secara otomatis.

## 2. Optimizer & Regularisasi
- **Optimizer:** `AdamW` ($lr = 10^{-3}$, `weight_decay` $= 10^{-4}$) untuk menjaga stabilitas pembaruan bobot saat proses *fine-tuning*.
- **Regularisasi:** Menggunakan *Dropout* ($p = 0.2$) sebelum lapisan proyeksi akhir serta *weight decay* pada `AdamW` untuk mencegah *overfitting* pada penutur latih.

## 3. Hyperparameter & Ablation Study
- **Status:** *Menunggu data primer (12 Nov)*. Eksperimen perbandingan rasio *freeze* lapisan, ukuran resolusi input spektrogram, dan *learning rate* akan dilakukan setelah data meja primer tersedia.

## 4. Catatan Pengujian Simulasi
- Skrip `python models/122450021/run.py --seed 1` telah diuji menggunakan data simulasi melalui `src.protokol.data.muat()` (pemisahan per penutur) dan dievaluasi lewat `src.common.evaluate`.
- **Catatan Penting:** Hasil pada `metrics_sintetis.json` merupakan validasi alur eksekusi data simulasi semata dan **bukan** capaian performa akhir model.
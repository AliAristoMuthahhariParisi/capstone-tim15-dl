# Catatan Desain Model: Autoencoder (Jalur AE)
**Nama:** Nayla Salsabila Fathianisa
**NIM:** 123450082 

### 1. Arsitektur Model
- **Model yang digunakan:** Small U-Net Autoencoder (CNN 2D).
- **Alasan pemilihan:** Model ini menggunakan struktur *Encoder-Decoder*, terdiri dari 2 lapis konvolusi untuk menyusutkan data, dan 2 lapis dekonvolusi untuk mengembalikannya ke bentuk semula. Desain ini dipilih karena cukup ringan namun tetap efisien untuk mengenali pola penting dari gambar spektrogram berdurasi 4 detik. Nantinya, fitur yang berhasil disusutkan ini bisa digunakan untuk mengukur skor anomali.

### 2. Optimizer dan Fungsi Loss
- **Optimizer:** Menggunakan AdamW dengan *learning rate* 1e-3 dan *weight decay* 1e-4. Pengaturan ini dipilih agar proses pelatihan berjalan stabil dan model tidak cepat menghafal data latih (*overfitting*).
- **Loss Function:** Menggunakan *Mean Squared Error* (MSE). Autoencoder dilatih untuk membangun ulang gambar agar semirip mungkin dengan aslinya. Oleh karena itu, MSE sangat cocok untuk menghitung jarak perbedaan antara hasil rekonstruksi dengan input awal.

### 3. Hyperparameter dan Regularisasi
- Pada Sprint 1 ini, pengaturan yang digunakan menggunakan standar awal, yaitu *learning rate* 1e-3 dan 50 *epoch*.
- Pencarian parameter terbaik (*hyperparameter tuning*) dan pengujian variasi *layer* lebih lanjut akan ditargetkan pada Sprint 2.

### 4. Hasil Evaluasi Data Primer (Sprint 1)
- Autoencoder bukan model pengklasifikasi, jadi metrik akurasi perintah bernilai 0.0 bukanlah ukuran yang tepat atau sebuah *error*. 
- Ukuran performa yang benar adalah SI-SDR, dan pada pengujian Seed 1 hasilnya **−0.76 dB**. Nilai negatif ini menunjukkan bahwa model belum berhasil merekonstruksi suara menjadi lebih bersih.
- Kemungkinan besar hal ini terjadi karena bising latar (seperti kipas/pompa) di rekaman sesi meja ini terlalu lemah atau data aslinya memang sudah sangat bersih, sehingga *autoencoder* yang dipaksa mencari dan membuang bising justru merusak sedikit sinyal aslinya.
- Penggunaan skema penilaian manual metrik di metrics.json ini bersifat sementara. Pembuatan *pipeline* penilaian metrik SI-SDR yang terpusat dan perbaikan strategi data masuk ke dalam target Sprint 2.
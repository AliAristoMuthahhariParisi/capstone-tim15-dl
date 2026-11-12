# Catatan Desain Model: Autoencoder (Jalur AE)
**Nama:** Nayla Salsabila Fathianisa
**NIM:** 123450082 

### 1. Arsitektur Model
- **Model yang digunakan:** Small U-Net Autoencoder (CNN 2D).
- **Alasan pemilihan:** Model ini menggunakan struktur *Encoder-Decoder* yang sederhana (2 lapis konvolusi untuk menyusutkan data, dan 2 lapis dekonvolusi untuk mengembalikannya ke bentuk semula). Desain ini dipilih karena cukup ringan namun tetap efisien untuk mengenali pola penting dari gambar spektrogram berdurasi 4 detik. Nantinya, fitur yang berhasil disusutkan ini bisa digunakan untuk mengukur skor anomali.

### 2. Optimizer dan Fungsi Loss
- **Optimizer:** Menggunakan AdamW dengan *learning rate* 1e-3 dan *weight decay* 1e-4. Pengaturan ini dipilih agar proses pelatihan berjalan stabil dan model tidak cepat menghafal data latih (*overfitting*).
- **Loss Function:** Menggunakan *Mean Squared Error* (MSE). Autoencoder dilatih untuk membangun ulang gambar agar semirip mungkin dengan aslinya. Oleh karena itu, MSE sangat cocok untuk menghitung jarak perbedaan antara hasil rekonstruksi dengan input awal.

### 3. Hyperparameter dan Regularisasi
- Untuk sementara, pengaturan yang digunakan masih bawaan standar (ukuran *batch* 32 dan 30 *epoch*).
- Pencarian parameter terbaik (*tuning*) dan pengujian lebih lanjut saat ini ditandai sebagai "menunggu data primer".

### 4. Hasil Simulasi
- Nilai metrik yang dihasilkan oleh skrip saat ini (yang menggunakan data buatan dari `jadwal.csv`) hanya bertujuan untuk memastikan bahwa seluruh alur program bisa berjalan lancar sampai akhir tanpa ada *error*.
- Nilai tersebut bukan pencapaian akhir dan belum bisa dijadikan patokan performa model ketika menggunakan data pengujian meja yang sebenarnya.
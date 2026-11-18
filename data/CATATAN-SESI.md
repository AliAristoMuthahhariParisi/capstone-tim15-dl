# Catatan Sesi Meja Eksperimen: Tim 15 (Ruang Cerdas 5 Volt)

## 1. Identitas Sesi dan Berkas

* **Kasus Tim:** Tim 15 — Pembersihan Ucapan dari Bising Mesin (*Speech Denoising from Machine Noise*)
* **ID Sesi:** `sesi-20261117-071715`
* **Waktu Pengambilan Data:** Selasa, 17 November 2026, pukul 07:17:15 s.d. 07:40:06 WIB
* **Tempat / Meja:** Meja Laboratorium Pengujian Ruang Cerdas 5 Volt
* **Host Pengendali:** `sditera-civ` (Python 3.12.3, perekam versi 1.13)
* **Penjaga Data & Evaluasi:** Lutfia Aisyah Putri (NIM 123450074)
* **Dosen Pembimbing:** Pak Ardika Satria
* **Sidik Data (`data_hash`):** `288d1dad9a827a418a79826a69f54389`

### Berkas yang Dikelola di `data/`
Berkas CSV dan manifes sesi meja ditempatkan pada folder `data/` sebagai acuan penghitungan `data_hash`:
1. `data/label.csv`: 240 baris kondisi pengujian (informasi partisipan, kalimat perintah, taraf aktuator kipas dan pompa, serta timestamp host).
2. `data/audio_index.csv`: Indeks 23 berkas rekaman audio kontinu 60 detik (`audio-0001.wav` s.d. `audio-0023.wav`).
3. `data/aktuator.csv`: Log status taraf aktuator kipas (PWM 0 atau 255) dan pompa (PWM 0 atau 200).
4. `data/daya.csv`: Log pembacaan sensor daya dan arus listrik meja.
5. `data/digital.csv`: Pembacaan status pin I/O digital.
6. `data/imu.csv`: Log sensor IMU/akselerometer meja laboratorium.
7. `data/lingkungan.csv`: Log sensor suhu udara (°C) dan iluminasi cahaya ruangan (lux).
8. `data/manifest_sesi.json`: Manifes asli dari perekam meja pengujian lab.
9. `data/MANIFEST.json`: Manifes starter untuk verifikasi keseragaman `data_hash` seluruh anggota tim.

> **Catatan Penyimpanan Berkas Audio:** Berkas audio mentah (`audio/*.wav`, total 23 berkas, ~138 MB) disimpan pada penyimpanan cloud tim dan disalin ke folder lokal `data/audio/`. Sesuai aturan repositori, berkas biner audio tidak di-commit ke Git.

---

## 2. Pemeriksaan Mutu Umum (Panduan Metode Bagian 8)

### 2.1 Kesesuaian Jumlah Kondisi
* **Rencana Eksperimen (`rencana/t15-ucapan-bersih.yaml`):** 240 kondisi.
* **Realisasi Data Meja (`data/label.csv`):** 240 kondisi (100% lengkap, tanpa ada baris yang hilang).
* **Rincian Per Partisipan:**
  Eksperimen melibatkan 6 partisipan (P01, P02, P03, P04, P05, P06).
  Masing-masing partisipan menjalankan tepat 40 kondisi pengujian:
  - 10 kondisi ucapan hening / bersih (taraf `kipas = 0`, `pompa = 0`).
  - 30 kondisi ucapan dengan bising mesin (kombinasi `kipas = 255, pompa = 0`, `kipas = 0, pompa = 200`, dan `kipas = 255, pompa = 200`).
  - Total kondisi hening seluruh sesi: 60 kondisi (25%).
  - Total kondisi bising mesin: 180 kondisi (75%).
  - Setiap kalimat perintah (10 kelas perintah) diucapkan tepat 4 kali oleh masing-masing partisipan.

### 2.2 Integritas Format dan Pembacaan Sensor
* Seluruh 7 berkas CSV sensor memiliki format baris yang konsisten, ber-timestamp sinkron dengan waktu host (`t_mulai_host` dan `t_selesai_host`).
* Frekuensi cuplik audio perekaman meja adalah 48 kHz (mono 16-bit PCM). Pipeline pemuat data melakukan *downsampling* terstandar ke 16 kHz untuk menyelaraskan dengan spesifikasi model.

---

## 3. Pemeriksaan Khusus Tim 15: Penjajaran Waktu Bersih dan Bising

Sesuai instruksi pemeriksaan khusus Tim 15:
> *Pasangan bersih dan bising sejajar waktunya; bila tidak, sejajarkan dengan korelasi silang.*

### 3.1 Metodologi Uji Korelasi Silang
Untuk menguji apakah terdapat pergeseran waktu (*temporal misalignment*) antara ucapan bersih dan ucapan bising, dilakukan analisis korelasi silang (*cross-correlation*) antar-sinyal:
1. Dipilih rekaman ucapan hening (`kipas = 0, pompa = 0`) sebagai sinyal referensi ucapan bersih.
2. Dipilih rekaman ucapan bising untuk kalimat yang sama dari partisipan yang sama.
3. Fungsi korelasi silang dinormalisasi dihitung untuk mencari titik puncak keterlambatan waktu (*time-lag peak*).

### 3.2 Hasil Evaluasi Penjajaran
* **Pergeseran Waktu Rata-rata (*Mean Time Lag*):** **-5,81 milidetik** (-93 sampel pada laju cuplik 16 kHz).
* **Kesimpulan Penjajaran:**
  Karena pergeseran waktu sangat kecil (< 10 ms) dan berada di bawah ukuran 1 frame hop STFT protokol tim (hop 16 ms / 256 sampel), sinyal rekaman meja berada dalam kondisi **sejajar secara presisi**. Penjajaran ulang waktu (*re-alignment*) manual tidak diperlukan karena waktu picu layar dan perekam telah sinkron secara otomatis.

---

## 4. Log Catatan Operator Sesi dan Kebijakan Penanganan Data

Berdasarkan lembar observasi operator meja laboratorium (`CATATAN SESI - CATATAN SESI.pdf`), terdapat 19 kondisi yang terdokumentasi mengalami gangguan akustik eksternal atau anomali partisipan:

| No. Kondisi | Partisipan | Kalimat Perintah | Catatan Gangguan Operator Meja | Status Data |
| :---: | :---: | :--- | :--- | :--- |
| **2** | P06 | kipas lebih pelan | **Tidak Ada Suara** (partisipan terlambat berbicara) | Dipertahankan (Anomali Uji) |
| **14** | P06 | matikan lampu meja | Suara Langkah Kaki | Dipertahankan |
| **19** | P06 | matikan lampu meja | Suara Batuk | Dipertahankan |
| **23** | P06 | nyalakan kipas sekarang | Suara Pintu Terbuka | Dipertahankan |
| **30** | P06 | matikan lampu meja | Suara Pintu Terbuka | Dipertahankan |
| **33** | P06 | batalkan perintah tadi | Suara Langkah Kaki | Dipertahankan |
| **34** | P06 | pompa berhenti | Suara Pintu Terbuka | Dipertahankan |
| **38** | P06 | batalkan perintah tadi | Suara Batuk | Dipertahankan |
| **40** | P06 | lampu lebih terang | Suara Langkah Kaki | Dipertahankan |
| **43** | P05 | meja siap dipakai | Suara Pintu Terbuka | Dipertahankan |
| **49** | P05 | pompa berhenti | Suara Langkah Kaki | Dipertahankan |
| **50** | P05 | matikan lampu meja | Suara Bersin | Dipertahankan |
| **60** | P05 | pompa berhenti | Suara Pintu Terbuka | Dipertahankan |
| **61** | P05 | matikan lampu meja | Suara Alarm HP | Dipertahankan |
| **71** | P05 | matikan lampu meja | Suara Pintu Terbuka | Dipertahankan |
| **78** | P05 | meja siap dipakai | Suara Langkah Kaki | Dipertahankan |
| **80** | P05 | matikan lampu meja | Suara Pintu Terbuka | Dipertahankan |
| **81** | P04 | batalkan perintah tadi | Suara Langkah Kaki | Dipertahankan |
| **98** | P04 | matikan lampu meja | Suara Batuk | Dipertahankan |

### 4.1 Kebijakan Data yang Dikeluarkan vs Dipertahankan
* **Kebijakan Seluruh Kondisi (240 Sampel):**
  Seluruh 240 kondisi **tetap dipertahankan** dalam dataset utama (`label.csv`) dan modul pemuat `src.protokol.data.muat()`.
* **Alasan Kebijakan:**
  1. Mempertahankan keseimbangan ukuran matriks data dan skema partisi penutur independen (*speaker-disjoint split*) yang telah disepakati seluruh anggota tim.
  2. Kondisi gangguan ruang (langkah kaki, pintu terbuka, batuk) mencerminkan bising lingkungan akustik nyata (*real-world acoustic noise*) di luar bising mesin, yang menguji ketahanan model penapis ucapan (*speech denoising robustness*).
  3. Khusus Kondisi No. 2 ("Tidak Ada Suara" pada partisipan P06), sampel ini berada pada **set uji (test set)**. Keberadaannya dicatat sebagai kasus uji batas (*corner case*) untuk memverifikasi apakah model mampu mengenali masukan hening tanpa menghasilkan prediksi perintah palsu (*false positive*).

---

## 5. Protokol Pemuatan dan Ekstraksi Data Tim

Modul `src/protokol/data.py` telah diperbarui untuk mengonsumsi data sesi meja ini secara otomatis:

1. **Pemotongan Berbasis Indeks Audio:**
   Audio 60 detik dipotong berdasarkan waktu `t_mulai_host` dan durasi 4,0 detik menggunakan referensi `data/audio_index.csv`.
2. **Penurunan Frekuensi Cuplik (*Downsampling*):**
   Sinyal audio 48 kHz diturunkan menjadi 16 kHz (`[::3]`), menghasilkan 64.000 sampel per kondisi 4 detik.
3. **Pembuangan Transien Awal:**
   Fungsi `src.protokol.pemotong.potong()` membuang 1,0 detik pertama (transien transisi relai dan motor kipas/pompa), menghasilkan 3,0 detik efektif (48.000 sampel @ 16 kHz).
4. **Ekstraksi Spektrogram:**
   Dihitung spektrogram log-magnitudo STFT berdimensi `(32 frame, 40 bin frekuensi)`.
5. **Pemisahan Partisipan Bebas Bocor (*Speaker-Disjoint Split*):**
   * **Data Latih (Train):** P01, P02, P03, P04 (160 sampel, 4 partisipan).
   * **Data Validasi (Val):** P05 (40 sampel, 1 partisipan).
   * **Data Uji (Test):** P06 (40 sampel, 1 partisipan).
   * Verifikasi otomatis `periksa_anti_bocor()` menjamin tidak ada tumpang tindih penutur antar-set.
6. **Normalisasi Standar:**
   Nilai mean dan standar deviasi dihitung hanya dari `X_train` per bin frekuensi, lalu diterapkan secara deterministik ke data validasi dan uji.

---

## 6. Panduan untuk Anggota Tim (Konsistensi Data)

Agar seluruh anggota tim menghasilkan nilai `data_hash` yang identik (`288d1dad9a827a418a79826a69f54389`) saat menjalankan pelatihan dan evaluasi artefak:

1. Tarik cabang `main` terbaru setelah PR kartu #5 digabungkan:
   ```bash
   git pull origin main
   ```
2. Pastikan berkas CSV berikut berada di folder `data/` lokal masing-masing:
   * `data/aktuator.csv`
   * `data/audio_index.csv`
   * `data/daya.csv`
   * `data/digital.csv`
   * `data/imu.csv`
   * `data/label.csv`
   * `data/lingkungan.csv`
3. Salin berkas audio ke `data/audio/*.wav`.
4. Jalankan verifikasi integritas manifes:
   ```bash
   python -c "from src.common.manifest import data_hash; print('data_hash:', data_hash())"
   ```
   Pastikan keluaran mencetak:
   `data_hash: 288d1dad9a827a418a79826a69f54389`

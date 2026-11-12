# Catatan Pengembangan Model — Lisa Diani Amelia (122450021)

## Identitas & Peran
- **Nama:** Lisa Diani Amelia
- **NIM:** 122450021
- **Peran & Fokus Model:** Transfer Learning — Memanfaatkan bobot model *pre-trained* untuk menaksir pola masker bising secara spasial pada gambar spektrogram agar ekstraksi fitur lebih efisien.
- **Peran Tambahan:** Integrator Repositori Tim (Kartu #6)
- **Kartu Terkait:** #6, #7, #17, #18


## Strategi & Rancangan Arsitektur

1. **Representasi Input (Spektrogram Spasial):**
   - Mengubah sinyal/fitur akustik menjadi representasi gambar spektrogram 2D (frekuensi $\times$ waktu) dan menyesuaikannya ke format 3 kanal (RGB/replikasi kanal) serta ukuran input standar model *pre-trained* (misalnya $224 \times 224$ piksel).
2. **Ekstraksi Fitur *Pre-trained* (Backbone):**
   - Memanfaatkan bobot arsitektur CNN ringan yang sudah dilatih sebelumnya (seperti **MobileNetV2** atau **ResNet18**) dengan membekukan (*freeze*) lapisan awal *feature extractor*.
   - Lapisan konvolusi *pre-trained* bertugas menangkap pola tekstur dan struktur spasial lokal pada spektrogram untuk memisahkan pola sinyal ucapan utama dari pola derau/masker bising latar belakang (seperti kipas/pompa).
3. **Klasifikasi Akhir (*Classifier Head*):**
   - Menambahkan lapisan *Global Average Pooling* dan *Linear/Dense Classifier* untuk memetakan representasi fitur spasial ke kelas kalimat target.


## Log Kemajuan & Eksperimen

### 1. Tahap Kerangka & Uji Integrasi Simulasi (Sprint 1 — 12 November 2026)
- **Status:** Siap Uji Integrasi (`run_all.py`)
- **Sumber Data:** Data simulasi 240 kondisi (`sesi-simulasi/tim-15-2026-11-12/jadwal.csv`) melalui `src/protokol/data.py`.
- **Implementasi Saat Ini (`run.py`):**
  - Menyiapkan *pipeline* ekstraksi fitur *pre-trained* (MobileNetV2) untuk spektrogram dengan mekanisme *fallback* aman jika dependensi *deep learning* belum terpasang di lingkungan uji.
  - Memastikan proses berjalan tuntas tanpa galat saat dipanggil oleh `run_all.py`.
  - Menyimpan metrik simulasi ke `results/122450021/metrics_sintetis.json` (tidak di-*commit* ke `metrics.json` sesuai aturan tim).

### 2. Rencana Iterasi Berikutnya (Data Riil & Fine-Tuning)
- Melatih *classifier head* dan melakukan *fine-tuning* sebagian blok konvolusi atas pada dataset riil (`data/label.csv`).
- Membandingkan efisiensi waktu inferensi dan performa (*Accuracy* serta *F1-Macro*) antara model *frozen backbone* vs *partial fine-tuning* terhadap baseline tim.
- Menyimpan hasil evaluasi resmi ke `results/122450021/metrics.json`.
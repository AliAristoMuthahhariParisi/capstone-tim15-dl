# Protokol evaluasi tim — Tim 15 (Ruang Cerdas 5 Volt)

Diisi Penjaga Data & Evaluasi (`123450074`), diajukan ke pembimbing lewat *pull request* ke `main`, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request* baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pembersihan ucapan dari bising mesin |
| Sumber dataset | Sesi simulasi (`sesi-simulasi/tim-15-2026-11-12/jadwal.csv`) untuk uji pipeline, dan rekaman meja uji 16 kHz (`audio/*.wav` + `label.csv`, 6 partisipan `P01`–`P06`) setelah slot meja |
| Jenis tugas (`TUGAS`) | `klasifikasi` |
| Target / label | 10 kelas kalimat perintah kendali alat meja uji (`0..9`) |
| Ukuran data latih / validasi / uji | `160 / 40 / 40` sampel (4 penutur latih, 1 penutur validasi, 1 penutur uji — total 240 kondisi) |
| Cara split | Pemisahan per penutur (*speaker-disjoint*): `P01`–`P04` latih, `P05` validasi, `P06` uji independen; dicek otomatis di `src/protokol/data.py` agar tidak ada penutur yang muncul di latih dan uji sekaligus |
| Metrik utama untuk komparasi | Peningkatan SI-SDR + akurasi perintah (serta macro-F1, ukuran ONNX dalam KB, dan latensi inferensi dalam ms) |
| Praproses yang sama untuk semua model | `src/protokol/pemotong.py` membuang 1,0 detik pertama tiap kondisi 4,0 detik (sisa 3,0 detik / 48.000 sampel @ 16 kHz); spektrogram log-magnitudo STFT (`N_FFT=512`, `32 frame × 40 bin`); normalisasi z-score dihitung **dari data latih saja** |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `a78f98074dbbce18cc8c05d5cd77708b` (diperbarui setelah data sesi meja resmi diterima) |

Disetujui pembimbing pada: … (tautan *pull request*)

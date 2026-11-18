# Protokol evaluasi tim — Tim 15 (Ruang Cerdas 5 Volt)

Diisi Penjaga Data & Evaluasi (`123450074`), diajukan ke pembimbing lewat *pull request* ke `main`, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request* baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pembersihan ucapan dari bising mesin |
| Sumber dataset | Data sesi meja resmi `sesi-20261117-071715` (`data/label.csv`, CSV sensor, dan `data/audio/*.wav`, 6 partisipan `P01`–`P06`, lihat `data/CATATAN-SESI.md`) serta fallback gladi simulasi |
| Jenis tugas (`TUGAS`) | `klasifikasi` |
| Target / label | 10 kelas kalimat perintah kendali alat meja uji (`0..9`) |
| Ukuran data latih / validasi / uji | `160 / 40 / 40` sampel (4 penutur latih, 1 penutur validasi, 1 penutur uji — total 240 kondisi) |
| Cara split | Pemisahan per penutur (*speaker-disjoint*): `P01`–`P04` latih, `P05` validasi, `P06` uji independen; dicek otomatis di `src/protokol/data.py` agar tidak ada penutur yang muncul di latih dan uji sekaligus |
| Metrik utama untuk komparasi | Peningkatan SI-SDR + akurasi perintah (serta macro-F1, ukuran ONNX dalam KB, dan latensi inferensi dalam ms) |
| Praproses yang sama untuk semua model | `src/protokol/pemotong.py` membuang 1,0 detik pertama tiap kondisi 4,0 detik (sisa 3,0 detik / 48.000 sampel @ 16 kHz); spektrogram log-magnitudo STFT (`N_FFT=512`, `32 frame × 40 bin`); normalisasi z-score dihitung **dari data latih saja** |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `288d1dad9a827a418a79826a69f54389` (data sesi meja resmi `sesi-20261117-071715`) |

Disetujui pembimbing pada: … (tautan *pull request*)

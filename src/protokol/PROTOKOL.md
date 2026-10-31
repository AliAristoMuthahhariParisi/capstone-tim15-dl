# Protokol evaluasi tim — Tim 15 (Ruang Cerdas 5 Volt)

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pembersihan ucapan dari bising mesin |
| Sumber dataset | Rekaman meja uji mikrofon 16 kHz (6 partisipan P01–P06) & rekaman bising mesin kipas/pompa |
| Jenis tugas (`TUGAS`) | klasifikasi |
| Target / label | 10 kelas perintah kontrol meja uji (0..9) |
| Ukuran data latih / validasi / uji | 180 / 20 / 40 sampel (total 240 spektrogram magnitudo) |
| Cara split | Partisipan P06 sebagai data uji independen (GroupKFold per pembicara); split latih & validasi memakai `split.indeks` (seed 2026, porsi validasi 10%) |
| Metrik utama untuk komparasi | macro-F1 dan akurasi (serta peningkatan SI-SDR dB, ukuran ONNX dalam KB, dan latensi inferensi dalam ms) |
| Praproses yang sama untuk semua model | 1 detik pertama tiap kondisi dibuang; ekstraksi spektrogram magnitudo STFT 16 kHz (durasi efektif 3 detik, matriks [64 frekuensi, 94 waktu]); normalisasi standar dihitung dari data latih saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | a78f98074dbbce18cc8c05d5cd77708b |

Disetujui pembimbing pada: … (tautan *pull request*)

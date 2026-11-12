# Laporan Gladi Rencana (Kartu #9)

**Tim**: 15  
**Rencana**: `rencana/t15-ucapan-bersih.yaml`  
**Tanggal**: 2026-11-12  
**Sifat**: rekonstruksi jadwal dari aturan dokumen, bukan keluaran `meja.py`, tanpa audio.

## Verifikasi rencana
- Seed: `1015`
- Blok: 6 × 40 kondisi
- Total: **240 kondisi**
- Durasi kondisi: 4.0 detik + jeda 0.5 detik
- Estimasi sesi: 24.0 menit (termasuk pergantian blok)
- Pompa menyala: 8.0 menit · Kipas: 8.0 menit

## 10 kondisi pertama

| no | blok | partisipan | kalimat | kipas | pompa |
|---:|---:|---|---|---:|---:|
| 1 | 1 | P06 | meja siap dipakai | 255 | 0 |
| 2 | 1 | P06 | semua alat mati | 0 | 200 |
| 3 | 1 | P06 | lampu lebih terang | 255 | 200 |
| 4 | 1 | P06 | batalkan perintah tadi | 255 | 200 |
| 5 | 1 | P06 | semua alat mati | 255 | 200 |
| 6 | 1 | P06 | nyalakan pompa sebentar | 255 | 200 |
| 7 | 1 | P06 | matikan lampu meja | 0 | 200 |
| 8 | 1 | P06 | semua alat mati | 0 | 0 |
| 9 | 1 | P06 | meja siap dipakai | 255 | 200 |
| 10 | 1 | P06 | batalkan perintah tadi | 0 | 0 |

## Bahan fisik
- Formulir persetujuan 6 lembar
- Lembar sesi 2 lembar
- Selotip penanda posisi, 2 pulpen

## Yang belum
- `meja.py --simulasi` belum dijalankan (paket belum tersedia di laptop tim).
- Audio dan `label.csv` asli baru ada setelah slot.

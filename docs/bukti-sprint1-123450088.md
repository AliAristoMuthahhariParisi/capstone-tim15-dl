# Bukti Sprint 1 — Ali Aristo Muthahhari Parisi (123450088)

Jalur RNN. Disusun 28 Nov 2026.

Semua angka di bawah dihasilkan ulang dari repo ini pada 28 Nov 2026, bukan disalin dari catatan.

---

## Kartu #13 — log menjalankan perintah + bentuk masukan/keluaran

**Perintah:** `python models/123450088/run.py --seed 1`
**Log lengkap:** `results/123450088/log-run-seed1.txt`
**Status:** selesai, exit code 0

```
sumber data : sesi asli: data
masukan     : latih (160, 32, 40), validasi (40, 32, 40)
model       : RNN LSTM atas baris frame (27,851 parameter, 10 kelas)
  epoch  1/60  loss=2.3063  val_acc=0.1000
  epoch 10/60  loss=2.2890  val_acc=0.0750
  epoch 20/60  loss=2.2074  val_acc=0.0750
  epoch 30/60  loss=1.9684  val_acc=0.1250
  epoch 40/60  loss=1.7005  val_acc=0.1000
  epoch 50/60  loss=1.4922  val_acc=0.1500
  epoch 60/60  loss=1.2469  val_acc=0.1250
```

**Bentuk masukan.** Spektrogram magnitudo 3D `[N, frame, bin]`: latih `(160, 32, 40)`,
validasi `(40, 32, 40)`. Dijaga oleh `assert Xtr.ndim == 3` di `run.py` baris 61 — run
berhenti bila dimensi tidak sesuai.

**Bentuk keluaran.** Logits `[N, 10]` = `(40, 10)` untuk 10 kelas perintah. Dijaga oleh
`assert logits.shape == (len(Xva), n_kelas)` di `run.py` baris 100.

**Arsitektur.** LSTM(40→64) atas 32 langkah waktu, kepala masker per frame
(`Linear(64,1)` + Sigmoid) yang mengalikan state tersembunyi, lalu rata-rata waktu dan
klasifikasi 10 kelas dengan dropout 0,3. Total 27.851 parameter.

### Run mode simulasi (30 epoch)

Log: `results/123450088/log-simulasi-30ep.txt`

```
sumber data : jadwal simulasi: tim-15-2026-11-12
masukan     : latih (160, 32, 40), validasi (40, 32, 40)
model       : RNN LSTM atas baris frame (27,851 parameter, 10 kelas)
  epoch  1/30  loss=2.3140  val_acc=0.1000
  epoch 10/30  loss=2.2589  val_acc=0.2500
  epoch 20/30  loss=1.8439  val_acc=0.4250
  epoch 30/30  loss=1.2620  val_acc=0.7000
{"val_acc": 0.7, "val_macro_f1": 0.661378} → results/123450088/metrics_sintetis.json
```

Hasilnya **identik** dengan artefak yang sudah tersimpan di repo (`metrics_sintetis.json`,
commit `946ec2a`: `acc 0.7`, `macro_f1 0.661378`), jadi run dapat direproduksi.

Cara menjalankan: pada salinan repo tanpa folder data sesi, dengan `data/MANIFEST.json`
sementara yang menunjuk `sesi-simulasi/tim-15-2026-11-12/jadwal.csv` agar penilai dapat
menghitung `data_hash`. Pada working tree yang sudah berisi data sesi, perintah yang sama
memilih data primer — log versi itu ada di `results/123450088/log-run-seed1.txt`.

---

## Kartu #14 — pemeriksaan artefak dan kecocokan data_hash

### 1. Skema berkas hasil

`results/123450088/metrics.json`:

| Kunci wajib | Status | Nilai |
|---|---|---|
| `metrik` | ADA | `val_acc` 0,125 · `val_macro_f1` 0,087273 |
| `seed` | ADA | 1 |
| `data_hash` | ADA | `a565707907851999be537f7e1272a55d` |
| `n_val` | ADA | 40 |

### 2. Kecocokan data_hash dengan MANIFEST

Berkas yang tercatat di `data/MANIFEST.json`: `data/aktuator.csv`, `data/audio_index.csv`,
`data/daya.csv`, `data/digital.csv`, `data/imu.csv`, `data/label.csv`, `data/lingkungan.csv`.

`data_hash` dihitung ulang dengan `src.common.manifest.data_hash()` + split beku
(`split:2026`), persis seperti `src/common/evaluate.py` baris 94:

```
data_hash segar   : a565707907851999be537f7e1272a55d
data_hash metrics : a565707907851999be537f7e1272a55d
HASIL             : COCOK
```

### 3. Berkas hasil tidak disunting tangan

`git status --porcelain -- results/123450088/metrics.json` → kosong (identik dengan commit).
Angka di `metrics.json` berasal dari `src.common.evaluate`, bukan diketik manual.

### 4. Catatan

`data/CATATAN-SESI.md` masih mencantumkan `data_hash` lama `288d1dad9a827a418a79826a69f54389`.
Nilai yang berlaku sekarang dan dipakai semua jalur adalah `a565707907851999be537f7e1272a55d`.
Dokumen itu perlu diperbarui oleh Penjaga Data agar tidak terbaca sebagai ketidakcocokan.

---

## Kartu #9 — paket simulasi, berbagi folder, kesiapan bahan

### 1. Paket simulasi di repo

Folder `sesi-simulasi/tim-15-2026-11-12/` berisi `jadwal.csv`, `metadata.json`,
`rencana.yaml`, dan `README.md`. Commit: `31582e5` — "gladi: jadwal 240 kondisi + laporan
kartu #9".

Verifikasi ulang 28 Nov 2026:

| Butir | Nilai |
|---|---|
| Jumlah kondisi | **240** (cocok dengan `metadata.json`) |
| Komposisi | 6 blok × 40 kondisi |
| Estimasi sesi | 24,0 menit |
| Pompa menyala | 8,0 menit |
| Kipas menyala | 8,0 menit |
| Tanggal gladi | 2026-11-12 |
| sha256 `jadwal.csv` | `887e75b9972474e3…` |

### 2. Berbagi folder ke Penjaga Data

Paket inilah yang dipakai Lutfia Aisyah Putri (123450074) untuk kartu #4 (pemotong jendela
dan `PROTOKOL.md`), dan kartu #4 sudah **disetujui** pada 28 Nov 11.51. Artinya paket sesi
simulasi sudah diterima dan dipakai oleh Penjaga Data.

### 3. Kesiapan bahan slot

Tercatat di `docs/hasil-gladi-simulasi.md` bagian "Bahan fisik": formulir persetujuan
6 lembar, lembar sesi 2 lembar, selotip penanda posisi, 2 pulpen.

### 4. Bagaimana simulasi dijalankan

Skrip yang tersedia di repo adalah `src/gladi_simulasi.py`, yang merekonstruksi jadwal dari
`rencana/t15-ucapan-bersih.yaml` dan memverifikasi bahwa rencana menghasilkan 240 kondisi
dengan komposisi benar. Skrip ini **bukan** `meja.py` dan tidak menghasilkan audio —
sifatnya rekonstruksi jadwal.

`python meja.py --simulasi` belum dapat dijalankan karena `meja.py` tidak ada di repo tim
(`git ls-files meja.py` kosong); alat itu disediakan pengelola meja. Keterbatasan ini juga
tercatat di `docs/hasil-gladi-simulasi.md` bagian "Yang belum".

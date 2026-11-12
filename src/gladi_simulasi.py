"""Gladi rencana tim 15 — rekonstruksi jadwal kondisi.

Kartu #9 (Gladi rencana: mode simulasi dan persiapan bahan slot).

CATATAN PENTING
Skrip ini BUKAN `meja.py` dan bukan pengganti `python meja.py --simulasi`.
Yang dihasilkan hanya JADWAL kondisi (urutan, taraf aktuator, waktu relatif),
tanpa audio dan tanpa label.csv. Gunanya:
  1. memastikan rencana menghasilkan 240 kondisi dengan komposisi yang benar;
  2. memberi Penjaga Data (#4) gambaran urutan dan waktu untuk menulis pemotong jendela;
  3. menghitung estimasi durasi sesi dan lama pompa menyala.

Validasi bahwa `meja.py` benar-benar menerima berkas rencana tetap harus
dilakukan di laptop meja sebelum slot.

Pakai:
    python src/protokol/gladi_simulasi.py [berkas_rencana.yaml]
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

AKAR = Path(__file__).resolve().parents[1]
RENCANA_DEFAULT = AKAR / "rencana" / "t15-ucapan-bersih.yaml"
GANTI_BLOK_DETIK = 60


def nilai_taraf(spek):
    if isinstance(spek, list):
        return spek
    if isinstance(spek, dict) and isinstance(spek.get("taraf"), list):
        return spek["taraf"]
    raise ValueError(f"Faktor tanpa daftar taraf: {spek!r}")


def pisahkan_faktor(faktor):
    manual, isyarat, otomatis = {}, {}, {}
    for nama, spek in faktor.items():
        if isinstance(spek, dict) and "jenis" in spek:
            jenis = str(spek["jenis"]).strip().lower()
            if jenis == "manual":
                manual[nama] = spek
            elif jenis == "isyarat":
                isyarat[nama] = spek
            else:
                otomatis[nama] = spek
        else:
            otomatis[nama] = spek
    return manual, isyarat, otomatis


def susun_jadwal(rencana):
    manual, isyarat, otomatis = pisahkan_faktor(rencana["faktor"])
    ulangan = int(rencana.get("ulangan", 1))
    ulangan_blok = int(rencana.get("ulangan_blok", 1))
    durasi = float(rencana["durasi_kondisi_detik"])
    jeda = float(rencana.get("jeda_detik", 0.5))

    nama_manual = list(manual)
    kombinasi_manual = list(itertools.product(*[nilai_taraf(manual[n]) for n in nama_manual]))
    nama_otomatis = list(otomatis)
    kombinasi_otomatis = list(itertools.product(*[nilai_taraf(otomatis[n]) for n in nama_otomatis]))
    nama_isyarat = list(isyarat)
    kombinasi_isyarat = list(itertools.product(*[nilai_taraf(isyarat[n]) for n in nama_isyarat]))

    satu_blok = [
        dict(zip(nama_otomatis, ao)) | dict(zip(nama_isyarat, si))
        for ao, si in itertools.product(kombinasi_otomatis, kombinasi_isyarat)
    ]

    rng = random.Random(int(rencana["seed"]))
    daftar_blok = kombinasi_manual * ulangan_blok
    rng.shuffle(daftar_blok)

    jadwal, t = [], 0.0
    no = 1
    for blok_ke, kombinasi in enumerate(daftar_blok, 1):
        kunci = dict(zip(nama_manual, kombinasi))
        isi = [kunci | k for k in satu_blok * ulangan]
        rng.shuffle(isi)
        t += GANTI_BLOK_DETIK
        for kondisi in isi:
            jadwal.append({
                "no": no,
                "blok": blok_ke,
                **kondisi,
                "mulai_detik": round(t, 1),
                "selesai_detik": round(t + durasi, 1),
            })
            t += durasi + jeda
            no += 1

    return jadwal, len(satu_blok) * ulangan, len(daftar_blok), durasi, jeda, t


def jalankan(berkas_rencana=RENCANA_DEFAULT):
    berkas_rencana = Path(berkas_rencana)
    if not berkas_rencana.is_absolute():
        berkas_rencana = AKAR / berkas_rencana
    rencana = yaml.safe_load(berkas_rencana.read_text(encoding="utf-8"))

    jadwal, per_blok, jumlah_blok, durasi, jeda, total_detik = susun_jadwal(rencana)
    total = len(jadwal)
    diharapkan = per_blok * jumlah_blok

    pompa_aktif = sum(1 for k in jadwal if k.get("pompa", 0) != 0) * durasi
    kipas_aktif = sum(1 for k in jadwal if k.get("kipas", 0) != 0) * durasi

    print("=" * 70)
    print("GLADI RENCANA TIM 15 — REKONSTRUKSI JADWAL (bukan meja.py)")
    print("=" * 70)
    print(f"Rencana   : {berkas_rencana.relative_to(AKAR) if AKAR in berkas_rencana.parents else berkas_rencana}")
    print(f"Seed      : {rencana['seed']}")
    print(f"Blok      : {jumlah_blok} × {per_blok} kondisi")
    print(f"Total     : {total} kondisi (diharapkan {diharapkan})")
    print(f"Per kondisi: {durasi} detik + jeda {jeda} detik")
    print(f"Durasi    : {total_detik/60:.1f} menit (termasuk {GANTI_BLOK_DETIK} detik pergantian blok)")
    print(f"Pompa     : {pompa_aktif/60:.1f} menit menyala")
    print(f"Kipas     : {kipas_aktif/60:.1f} menit menyala")

    if total != diharapkan:
        print("GAGAL: jumlah kondisi tidak sesuai rencana. Berkas tidak ditulis.")
        return 1

    keluaran = AKAR / "sesi-simulasi" / f"tim-15-{date.today().isoformat()}"
    keluaran.mkdir(parents=True, exist_ok=True)

    kolom = ["no", "blok"] + [k for k in jadwal[0] if k not in ("no", "blok", "mulai_detik", "selesai_detik")] + ["mulai_detik", "selesai_detik"]
    with (keluaran / "jadwal.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        w.writerows(jadwal)

    shutil.copyfile(berkas_rencana, keluaran / "rencana.yaml")

    metadata = {
        "tim": rencana.get("tim"),
        "judul": rencana.get("judul"),
        "seed": rencana["seed"],
        "total_kondisi": total,
        "kondisi_per_blok": per_blok,
        "jumlah_blok": jumlah_blok,
        "durasi_kondisi_detik": durasi,
        "jeda_detik": jeda,
        "ganti_blok_detik": GANTI_BLOK_DETIK,
        "estimasi_total_menit": round(total_detik / 60, 1),
        "pompa_menyala_menit": round(pompa_aktif / 60, 1),
        "kipas_menyala_menit": round(kipas_aktif / 60, 1),
        "tanggal_gladi": date.today().isoformat(),
        "sumber": "rekonstruksi aturan dokumen (bukan keluaran meja.py); tanpa audio, tanpa label.csv",
    }
    (keluaran / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (keluaran / "README.md").write_text(
        "# Paket Sesi Simulasi — Tim 15\n\n"
        "Isi: `jadwal.csv` (urutan kondisi), `metadata.json`, `rencana.yaml` (salinan).\n\n"
        "Ini **jadwal hasil rekonstruksi**, bukan keluaran `meja.py --simulasi`, dan tidak memuat audio.\n"
        "Dipakai untuk menguji pemotong jendela dan `run.py` sebelum data meja tersedia.\n"
        "Validasi bahwa `meja.py` menerima rencana ini tetap dilakukan di laptop meja sebelum slot.\n",
        encoding="utf-8",
    )

    laporan = AKAR / "docs" / "hasil-gladi-simulasi.md"
    laporan.parent.mkdir(parents=True, exist_ok=True)
    baris_contoh = "\n".join(
        f"| {k['no']} | {k['blok']} | {k.get('partisipan','-')} | {k.get('kalimat','-')} | {k.get('kipas','-')} | {k.get('pompa','-')} |"
        for k in jadwal[:10]
    )
    laporan.write_text(
        "# Laporan Gladi Rencana (Kartu #9)\n\n"
        f"**Tim**: 15  \n**Rencana**: `rencana/t15-ucapan-bersih.yaml`  \n"
        f"**Tanggal**: {date.today().isoformat()}  \n"
        "**Sifat**: rekonstruksi jadwal dari aturan dokumen, bukan keluaran `meja.py`, tanpa audio.\n\n"
        "## Verifikasi rencana\n"
        f"- Seed: `{rencana['seed']}`\n- Blok: {jumlah_blok} × {per_blok} kondisi\n- Total: **{total} kondisi**\n"
        f"- Durasi kondisi: {durasi} detik + jeda {jeda} detik\n"
        f"- Estimasi sesi: {total_detik/60:.1f} menit (termasuk pergantian blok)\n"
        f"- Pompa menyala: {pompa_aktif/60:.1f} menit · Kipas: {kipas_aktif/60:.1f} menit\n\n"
        "## 10 kondisi pertama\n\n| no | blok | partisipan | kalimat | kipas | pompa |\n|---:|---:|---|---|---:|---:|\n"
        f"{baris_contoh}\n\n"
        "## Bahan fisik\n- Formulir persetujuan 6 lembar\n- Lembar sesi 2 lembar\n- Selotip penanda posisi, 2 pulpen\n\n"
        "## Yang belum\n- `meja.py --simulasi` belum dijalankan (paket belum tersedia di laptop tim).\n"
        "- Audio dan `label.csv` asli baru ada setelah slot.\n",
        encoding="utf-8",
    )

    print(f"\n[OK] jadwal  : {keluaran.relative_to(AKAR)}/jadwal.csv")
    print(f"[OK] metadata : {keluaran.relative_to(AKAR)}/metadata.json")
    print(f"[OK] laporan  : {laporan.relative_to(AKAR)}")
    print("\nLangkah berikut: (1) buka PR, (2) serahkan folder ke Penjaga Data, (3) minta pengelola menjalankan meja.py --simulasi sekali.")
    return 0


if __name__ == "__main__":
    sys.exit(jalankan(sys.argv[1] if len(sys.argv) > 1 else RENCANA_DEFAULT))

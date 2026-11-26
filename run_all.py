"""Satu perintah reproduksi tim.

    python run_all.py [--seeds 1 2 3] [--final] [--hanya <nim> ...] [--rakit-saja]

Menjalankan baseline dosen (``src.common.baseline``) dan ``models/<nim>/run.py`` setiap anggota
untuk tiap seed, lalu merakit ``reports/hasil_komparasi.md`` dari ``results/*/metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def muat_hasil_tersimpan(ringkas: dict[str, list[dict]], hanya: list[str] | None) -> None:
    """Memuat seluruh results/<nim>/metrics.json yang tersedia jika tidak dijalankan ulang."""
    for p in sorted((AKAR / "results").iterdir()):
        if not p.is_dir() or p.name == "baseline":
            continue
        if hanya is not None and p.name not in hanya:
            continue
        mj = p / "metrics.json"
        if mj.exists() and p.name not in ringkas:
            ringkas[p.name] = [json.loads(mj.read_text("utf-8"))]


def format_baris(nim: str, h: dict) -> str:
    """Memformat satu baris tabel komparasi dari struktur metrics.json yang fleksibel."""
    mt = h.get("metrik") or h.get("metrics") or {}
    model = h.get("model", "-")
    seed = str(h.get("seed", "-"))
    durasi = f"{h['durasi_detik']:.1f}" if "durasi_detik" in h else "—"
    perangkat = h.get("perangkat", "—")

    if "val_acc" in mt:
        val_acc = f"{mt['val_acc']:.4f}"
        val_f1 = f"{mt['val_macro_f1']:.4f}" if "val_macro_f1" in mt else "—"
    elif "akurasi_perintah" in mt:
        val_acc = f"{mt['akurasi_perintah']:.4f} (perintah)"
        val_f1 = f"SI-SDR {mt['si_sdr_peningkatan']:.2f} dB" if "si_sdr_peningkatan" in mt else "—"
    else:
        val_acc = "—"
        val_f1 = "—"

    test_str = f"{mt['test_acc']:.4f}" if "test_acc" in mt else "—"
    return f"| {nim} | {model} | {seed} | {val_acc} | {val_f1} | {test_str} | {durasi} | {perangkat} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None, help="NIM tertentu saja")
    ap.add_argument("--tanpa-baseline", action="store_true")
    ap.add_argument("--rakit-saja", action="store_true", help="Rakit tabel dari results/*/metrics.json tanpa melatih ulang")
    a = ap.parse_args()

    ringkas: dict[str, list[dict]] = {}

    if not a.rakit_saja:
        if not a.tanpa_baseline:
            jalankan([sys.executable, "-m", "src.common.baseline", "--seed", str(a.seeds[0])])
        anggota = sorted(
            p for p in (AKAR / "models").iterdir()
            if p.is_dir() and (p / "run.py").exists()
            and p.name != "TEMPLATE" and (a.hanya is None or p.name in a.hanya)
        )
        for m in anggota:
            for s in a.seeds:
                perintah = [sys.executable, str(m / "run.py"), "--seed", str(s)]
                if a.final and s == a.seeds[-1]:
                    perintah.append("--final")
                jalankan(perintah)
                hasil = json.loads((AKAR / "results" / m.name / "metrics.json").read_text("utf-8"))
                ringkas.setdefault(m.name, []).append(hasil)
    else:
        muat_hasil_tersimpan(ringkas, a.hanya)

    baris = [
        "| NIM | Model | seed | val_acc | val_macro_f1 | test_acc | durasi (s) | perangkat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    bl = AKAR / "results" / "baseline" / "metrics.json"
    if bl.exists():
        baris.append(format_baris("baseline", json.loads(bl.read_text("utf-8"))))

    for nim, daftar in ringkas.items():
        for h in daftar:
            baris.append(format_baris(nim, h))

    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text("# Hasil komparasi (dibangkitkan run_all.py)\n\n" + "\n".join(baris) + "\n", encoding="utf-8")
    print("tabel komparasi ->", out.relative_to(AKAR))


if __name__ == "__main__":
    main()

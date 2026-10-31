"""Pemuat data kasus Tim 15 (Pembersihan ucapan dari bising mesin).

Kasus: Pembersihan ucapan dari bising mesin (Ruang Cerdas 5 Volt).
Pembimbing: Ardika Satria.
Masukan: Spektrogram magnitudo STFT (16 kHz, 3 detik efektif, [N, 64, 94]).
Target: 10 kelas kalimat perintah kendali alat meja uji (0..9).

Pemisahan data:
- Latih & Validasi: Partisipan P01 s.d. P05, dibagi dengan split beku (seed 2026, porsi validasi 10%).
- Data Uji: Partisipan P06 sebagai hold-out test set (GroupKFold per partisipan/pembicara).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"
BERKAS_DATA = manifest.AKAR / "data" / "spektrogram_rekaman.npz"


def muat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test)."""
    if not BERKAS_DATA.exists():
        raise FileNotFoundError(
            f"Berkas dataset {BERKAS_DATA} tidak ditemukan di data/. "
            "Pastikan berkas spektrogram sudah ada dan terdaftar di MANIFEST.json."
        )

    arsip = np.load(BERKAS_DATA)
    X = arsip["X"].astype(np.float32)
    y = arsip["y"].astype(np.int64)
    partisipan = arsip["partisipan"]

    # Pemisahan data uji per partisipan (P06 sebagai data uji independen)
    mask_uji = (partisipan == "P06")
    mask_dev = ~mask_uji

    X_dev, y_dev = X[mask_dev], y[mask_dev]
    X_test, y_test = X[mask_uji], y[mask_uji]

    # Split latih dan validasi menggunakan split beku seed 2026
    idx_latih, idx_val = split.indeks(len(X_dev), porsi_val=0.1, seed=2026)

    X_train, y_train = X_dev[idx_latih], y_dev[idx_latih]
    X_val, y_val = X_dev[idx_val], y_dev[idx_val]

    return X_train, y_train, X_val, y_val, X_test, y_test

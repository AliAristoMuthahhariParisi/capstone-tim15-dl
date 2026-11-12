"""Pemotong jendela audio — Tim 15 (pembersihan ucapan dari bising mesin).

Aturan protokol:
    - setiap kondisi berdurasi 4,0 detik pada laju cuplik 16 kHz;
    - 1,0 detik pertama dibuang (transien akselerasi kipas/pompa & jeda awal bicara);
    - sisa 3,0 detik dipakai sebagai jendela masukan (dipotong atau di-pad tepat 3,0 detik).
"""
from __future__ import annotations

import numpy as np

SR_BAWAAN = 16000
BUANG_AWAL_DETIK = 1.0
DURASI_KONDISI_DETIK = 4.0
DURASI_EFEKTIF_DETIK = 3.0


def potong(
    sinyal: np.ndarray,
    sr: int = SR_BAWAAN,
    buang_awal: float = BUANG_AWAL_DETIK,
    durasi_target: float | None = DURASI_EFEKTIF_DETIK,
) -> np.ndarray:
    """Buang bagian awal sinyal, lalu potong atau pad hingga panjang target."""
    sinyal = np.asarray(sinyal, dtype=np.float32).ravel()
    n_buang = int(round(buang_awal * sr))
    sisa = sinyal[n_buang:] if len(sinyal) > n_buang else np.zeros(0, dtype=np.float32)

    if durasi_target is None:
        return sisa

    n_target = int(round(durasi_target * sr))
    if len(sisa) >= n_target:
        return sisa[:n_target]
    hasil = np.zeros(n_target, dtype=np.float32)
    hasil[: len(sisa)] = sisa
    return hasil


def potong_batch(
    daftar: list[np.ndarray],
    sr: int = SR_BAWAAN,
    buang_awal: float = BUANG_AWAL_DETIK,
    durasi_target: float | None = DURASI_EFEKTIF_DETIK,
) -> np.ndarray:
    """Potong sekumpulan sinyal menjadi satu array 2D [N, n_sampel]."""
    return np.stack([potong(s, sr, buang_awal, durasi_target) for s in daftar], axis=0)


if __name__ == "__main__":
    sr = SR_BAWAAN
    t = np.arange(int(DURASI_KONDISI_DETIK * sr)) / sr
    uji = 0.05 * np.random.default_rng(1).standard_normal(len(t)).astype(np.float32)
    ucapan = (t >= 1.4) & (t <= 2.2)
    uji[ucapan] += 0.3 * np.sin(2 * np.pi * 300 * t[ucapan]).astype(np.float32)

    hasil = potong(uji, sr=sr, durasi_target=DURASI_EFEKTIF_DETIK)
    print(f"masukan : {len(uji)} sampel ({len(uji) / sr:.2f} detik)")
    print(f"keluaran: {len(hasil)} sampel ({len(hasil) / sr:.2f} detik)")
    assert len(hasil) == int(DURASI_EFEKTIF_DETIK * sr), "hasil harus tepat 3 detik"
    assert np.allclose(hasil, uji[sr:]), "isi hasil harus sisa setelah 1 detik pertama"
    print("[OK] pemotong: 1 detik awal dibuang, sisa 3 detik valid")

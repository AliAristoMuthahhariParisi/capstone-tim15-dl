"""Pemuat data kasus Tim 15 — pembersihan ucapan dari bising mesin.

Kontrak starter (dipakai ``src.common.evaluate``):
    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

Aturan protokol tim yang ditegakkan di modul ini:
1. Pemotongan jendela: membuang 1,0 detik pertama tiap kondisi (sisa 3,0 detik)
   lewat ``src.protokol.pemotong``.
2. Ekstraksi fitur: spektrogram log-magnitudo STFT (32 frame x 40 bin frekuensi).
3. Pemisahan per penutur (Group split): penutur pada data latih, validasi, dan uji
   terpisah sepenuhnya (P01-P04 latih, P05 validasi, P06 uji) dan diverifikasi
   otomatis dengan assertion agar tidak ada satuan pemisahan yang bocor.
4. Normalisasi: rata-rata dan simpangan baku dihitung HANYA dari data latih
   (``X_train``), lalu diterapkan ke validasi dan uji.

Dua sumber data yang didukung:
- Folder sesi asli (``TIM15_SESI`` atau ``data/sesi/*/label.csv``) berisi ``audio/*.wav``.
- Jadwal simulasi gladi (``sesi-simulasi/tim-15-*/jadwal.csv``) untuk uji pipeline.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from src.common import split
from src.protokol import pemotong

AKAR = Path(__file__).resolve().parents[2]
TUGAS = "klasifikasi"

KALIMAT = [
    "nyalakan kipas sekarang",
    "matikan lampu meja",
    "pompa berhenti",
    "kipas lebih pelan",
    "lampu lebih terang",
    "semua alat mati",
    "nyalakan pompa sebentar",
    "kipas paling kencang",
    "batalkan perintah tadi",
    "meja siap dipakai",
]

SR = pemotong.SR_BAWAAN
N_FFT = 512
N_BIN = 40
N_FRAME = 32

_sumber = "belum dimuat"
_ringkasan_split: dict[str, list[str]] = {}


def sumber_data() -> str:
    return _sumber


def ringkasan_split() -> dict[str, list[str]]:
    """Kembalikan daftar penutur unik pada masing-masing bagian latih, validasi, dan uji."""
    return _ringkasan_split


# --------------------------------------------------------------------------- fitur
def ciri(sinyal: np.ndarray, sr: int = SR) -> np.ndarray:
    """Ekstrak spektrogram log-magnitudo (32 frame x 40 bin) dari jendela 3 detik."""
    x = np.asarray(sinyal, dtype=np.float32).ravel()
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    jendela = np.hanning(N_FFT).astype(np.float32)
    mulai = np.linspace(0, max(0, len(x) - N_FFT), N_FRAME).astype(int)
    bingkai = np.stack([x[i : i + N_FFT] * jendela for i in mulai])
    spek = np.abs(np.fft.rfft(bingkai, axis=1)).astype(np.float32)
    tepi = np.linspace(0, spek.shape[1], N_BIN + 1).astype(int)
    hasil = np.stack(
        [spek[:, tepi[b] : max(tepi[b] + 1, tepi[b + 1])].mean(axis=1) for b in range(N_BIN)],
        axis=1,
    )
    return np.log1p(hasil).astype(np.float32)


# --------------------------------------------------------------------------- simulasi jadwal
def _kalimat_jadi_sinyal(indeks: int, kipas: float, pompa: float, no_kondisi: int, sr: int = SR) -> np.ndarray:
    """Bangkitkan sinyal 4 detik untuk uji pipeline dari satu baris jadwal simulasi."""
    rng = np.random.default_rng(1000 + no_kondisi)
    n = int(pemotong.DURASI_KONDISI_DETIK * sr)
    t = np.arange(n) / sr
    x = np.zeros(n, dtype=np.float32)

    f0 = 110 + 20 * indeks
    durasi = 0.7 + 0.06 * len(KALIMAT[indeks])
    i0 = int(2.5 * sr)
    i1 = min(n, i0 + int(durasi * sr))
    tt = t[i0:i1] - t[i0]
    ucapan = sum(0.45 / (k + 1) * np.sin(2 * np.pi * f0 * (k + 1) * tt) for k in range(3))
    x[i0:i1] += ucapan.astype(np.float32) * np.hanning(i1 - i0).astype(np.float32)

    if kipas and float(kipas) > 0:
        x += (0.10 * rng.standard_normal(n) + 0.04 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    if pompa and float(pompa) > 0:
        x += (0.06 * rng.standard_normal(n) + 0.07 * np.sin(2 * np.pi * 70 * t)).astype(np.float32)
    return x


def _dari_jadwal(berkas: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baris = list(csv.DictReader(berkas.open(encoding="utf-8-sig")))
    X, y, grup = [], [], []
    for i, b in enumerate(baris):
        nama = str(b["kalimat"]).strip().lower()
        idx = KALIMAT.index(nama) if nama in KALIMAT else 0
        no_k = int(b.get("no", i + 1))
        sinyal = _kalimat_jadi_sinyal(idx, float(b.get("kipas", 0)), float(b.get("pompa", 0)), no_k)
        jendela_3d = pemotong.potong(sinyal, sr=SR, durasi_target=pemotong.DURASI_EFEKTIF_DETIK)
        X.append(ciri(jendela_3d, sr=SR))
        y.append(idx)
        grup.append(str(b.get("partisipan", "P01")).strip())
    return np.stack(X), np.asarray(y, np.int64), np.asarray(grup)


# --------------------------------------------------------------------------- sesi asli (.wav + label.csv)
def _folder_sesi() -> Path | None:
    env = os.environ.get("TIM15_SESI")
    if env and (Path(env) / "label.csv").exists():
        return Path(env)
    for kandidat in sorted((AKAR / "data" / "sesi").glob("*/label.csv")):
        return kandidat.parent
    return None


def _dari_sesi(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import wave

    label = list(csv.DictReader((folder / "label.csv").open(encoding="utf-8-sig")))
    if not label:
        raise ValueError(f"{folder}/label.csv kosong")

    kolom_kalimat = next((k for k in label[0] if "kalimat" in k.lower() or "kata" in k.lower()), None)
    if kolom_kalimat is None:
        raise ValueError("label.csv tidak memiliki kolom kalimat/kata")
    kolom_penutur = next((k for k in label[0] if "partisipan" in k.lower() or "penutur" in k.lower()), None)

    audio = sorted((folder / "audio").glob("*.wav"))
    if len(audio) != len(label):
        raise ValueError(f"jumlah berkas audio ({len(audio)}) != jumlah baris label ({len(label)})")

    X, y, grup = [], [], []
    for baris, berkas in zip(label, audio):
        with wave.open(str(berkas), "rb") as w:
            sr = w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                raw = raw.reshape(-1, w.getnchannels()).mean(axis=1)
        jendela_3d = pemotong.potong(raw, sr=sr, durasi_target=pemotong.DURASI_EFEKTIF_DETIK)
        X.append(ciri(jendela_3d, sr=sr))
        nama = str(baris[kolom_kalimat]).strip().lower()
        y.append(KALIMAT.index(nama) if nama in KALIMAT else 0)
        penutur = str(baris[kolom_penutur]).strip() if kolom_penutur else berkas.stem.split("_")[0]
        grup.append(penutur)
    return np.stack(X), np.asarray(y, np.int64), np.asarray(grup)


# --------------------------------------------------------------------------- split & normalisasi
def _pisah_per_penutur(grup: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pisahkan indeks latih, validasi, dan uji berdasarkan satuan penutur (disjoint)."""
    penutur_unik = sorted(set(grup.tolist()))
    if len(penutur_unik) < 3:
        raise ValueError(f"Minimal butuh 3 penutur untuk split latih/val/uji, ditemukan: {penutur_unik}")

    penutur_uji = {penutur_unik[-1]}          # mis. P06 untuk data uji
    penutur_val = {penutur_unik[-2]}          # mis. P05 untuk validasi
    penutur_latih = set(penutur_unik[:-2])    # mis. P01..P04 untuk latih

    idx_tr = np.where(np.isin(grup, list(penutur_latih)))[0]
    idx_va = np.where(np.isin(grup, list(penutur_val)))[0]
    idx_te = np.where(np.isin(grup, list(penutur_uji)))[0]
    return idx_tr, idx_va, idx_te


def periksa_anti_bocor(grup: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray, idx_te: np.ndarray) -> None:
    """Pastikan tidak ada satu pun penutur yang muncul di latih, validasi, atau uji sekaligus."""
    s_tr = set(grup[idx_tr].tolist())
    s_va = set(grup[idx_va].tolist())
    s_te = set(grup[idx_te].tolist())

    assert s_tr.isdisjoint(s_te), f"Kebocoran penutur antara latih dan uji: {s_tr & s_te}"
    assert s_tr.isdisjoint(s_va), f"Kebocoran penutur antara latih dan validasi: {s_tr & s_va}"
    assert s_va.isdisjoint(s_te), f"Kebocoran penutur antara validasi dan uji: {s_va & s_te}"


def _normalisasi_dari_latih(
    X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hitung mean & std hanya dari X_train (per bin frekuensi), terapkan ke semua split."""
    rata2 = X_tr.mean(axis=(0, 1), keepdims=True)
    simpangan = X_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((X_tr - rata2) / simpangan).astype(np.float32),
        ((X_va - rata2) / simpangan).astype(np.float32),
        ((X_te - rata2) / simpangan).astype(np.float32),
    )


# --------------------------------------------------------------------------- kontrak utama
def muat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test) terpisah per penutur."""
    global _sumber, _ringkasan_split

    folder = _folder_sesi()
    if folder is not None:
        X, y, grup = _dari_sesi(folder)
        _sumber = f"sesi asli: {folder.name}"
    else:
        kandidat = sorted((AKAR / "sesi-simulasi").glob("tim-15-*/jadwal.csv"))
        if not kandidat:
            raise FileNotFoundError(
                "Tidak ada data. Jalankan `python src/gladi_simulasi.py` dulu, "
                "atau set TIM15_SESI ke folder sesi hasil meja.py."
            )
        X, y, grup = _dari_jadwal(kandidat[-1])
        _sumber = f"jadwal simulasi: {kandidat[-1].parent.name}"

    idx_tr, idx_va, idx_te = _pisah_per_penutur(grup)
    periksa_anti_bocor(grup, idx_tr, idx_va, idx_te)

    _ringkasan_split = {
        "latih": sorted(set(grup[idx_tr].tolist())),
        "validasi": sorted(set(grup[idx_va].tolist())),
        "uji": sorted(set(grup[idx_te].tolist())),
    }

    X_tr, X_va, X_te = _normalisasi_dari_latih(X[idx_tr], X[idx_va], X[idx_te])
    return X_tr, y[idx_tr], X_va, y[idx_va], X_te, y[idx_te]


must = muat


if __name__ == "__main__":
    Xtr, ytr, Xva, yva, Xte, yte = muat()
    info = ringkasan_split()
    print(f"sumber      : {sumber_data()}")
    print(f"latih       : X {Xtr.shape} {Xtr.dtype} · y {ytr.shape} · penutur {info['latih']}")
    print(f"validasi    : X {Xva.shape} · y {yva.shape} · penutur {info['validasi']}")
    print(f"uji         : X {Xte.shape} · y {yte.shape} · penutur {info['uji']}")
    print(f"stat latih  : mean={float(Xtr.mean()):.4f}, std={float(Xtr.std()):.4f}")
    print("[OK] verifikasi anti-bocor penutur lulus (latih, validasi, uji terpisah sempurna)")

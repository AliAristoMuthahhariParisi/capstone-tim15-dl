"""Kerangka model jalur RNN (LSTM/GRU) — Tim 15, NIM 123450088.

Kartu #13: `python models/123450088/run.py --seed 1` berjalan sampai akhir pada data simulasi.

Arsitektur:
    Jaringan berulang (LSTM) atas urutan 32 frame spektrogram, dengan kepala
    penaksir masker per frame (mask time-frequency) dan klasifikasi 10 perintah.

Pemakaian:
    python models/123450088/run.py --seed 1 [--epochs 30] [--final]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "rnn"


def bangun_rnn(n_kelas: int):
    """LSTM atas urutan frame + kepala masker time-frequency dan klasifikasi."""
    import torch
    import torch.nn as nn

    class RNNMaskerSpektrogram(nn.Module):
        def __init__(self, jumlah_kelas: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=40, hidden_size=64, num_layers=1, batch_first=True)
            # kepala masker: satu nilai per frame (time mask)
            self.masker = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
            self.kepala = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64, jumlah_kelas),
            )

        def forward(self, x):
            # x: [B, frame, bin]
            urut, _ = self.lstm(x)
            bobot = self.masker(urut)              # [B, frame, 1]
            urut_bersih = urut * bobot
            ringkas = urut_bersih.mean(dim=1)      # [B, hidden]
            return self.kepala(ringkas)

    return RNNMaskerSpektrogram(n_kelas)


def latih_dan_prediksi(seed: int, final: bool, epochs: int = 30, batch_size: int = 32, lr: float = 1e-3):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    sumber = data.sumber_data()
    assert Xtr.ndim == 3, f"Masukan spektrogram harus 3D [N, frame, bin], ditemukan {Xtr.shape}"
    n_kelas = len(data.KALIMAT)

    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64), dtype=torch.long)
    Xva_t = torch.tensor(np.asarray(Xva, dtype=np.float32))

    model = bangun_rnn(n_kelas)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"sumber data : {sumber}")
    print(f"masukan     : latih {tuple(Xtr_t.shape)}, validasi {tuple(Xva_t.shape)}")
    print(f"model       : RNN LSTM atas baris frame ({n_param:,} parameter, {n_kelas} kelas)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rugi_fn = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        urutan = rng.permutation(len(Xtr_t))
        total = 0.0
        for i in range(0, len(urutan), batch_size):
            idx = urutan[i : i + batch_size]
            opt.zero_grad()
            keluar = model(Xtr_t[idx])
            rugi = rugi_fn(keluar, ytr_t[idx])
            rugi.backward()
            opt.step()
            total += float(rugi.item()) * len(idx)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred_ep = model(Xva_t).argmax(dim=1).numpy()
            print(f"  epoch {ep:2d}/{epochs}  loss={total/len(Xtr_t):.4f}  val_acc={(pred_ep == yva).mean():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xva_t)
        assert logits.shape == (len(Xva), n_kelas), f"Bentuk keluaran tidak sesuai: {logits.shape}"
        pred_val = logits.argmax(dim=1).numpy().astype(np.int64)
        pred_test = (
            model(torch.tensor(np.asarray(Xte, dtype=np.float32))).argmax(dim=1).numpy().astype(np.int64)
            if final and Xte is not None
            else None
        )
    return pred_val, pred_test, sumber


def tulis_metrik_cadangan(out: Path, pred_val, y_val, seed: int, durasi: float, perangkat: str, sumber: str) -> None:
    """Dipakai bila `src.common.evaluate` tidak bisa jalan (mis. berkas MANIFEST belum ada).

    Menulis ke metrics_sintetis.json, bukan metrics.json, dan menandai sumber datanya.
    """
    from src.common.evaluate import metrik_klasifikasi

    Xtr, _, Xva, yva, _, _ = data.muat()
    sidik = hashlib.sha256(np.asarray(Xtr).tobytes() + np.asarray(Xva).tobytes()).hexdigest()[:32]
    metrik = metrik_klasifikasi(np.asarray(yva), np.asarray(pred_val))
    hasil = {
        "metrik": {k: round(float(v), 6) for k, v in metrik.items()},
        "seed": int(seed),
        "data_hash": sidik,
        "durasi_detik": round(durasi, 1),
        "perangkat": perangkat,
        "model": JALUR,
        "n_val": int(len(yva)),
        "n_test": 0,
        "sumber_data": sumber,
        "catatan": "metrik dari data sintetis; berkas data/MANIFEST belum tersedia di mesin ini",
    }
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(hasil["metrik"]), "->", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kerangka jalur RNN Tim 15")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test, sumber = latih_dan_prediksi(a.seed, a.final, epochs=a.epochs)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    # metrik pada data simulasi ditulis terpisah, tidak menimpa metrics.json resmi
    if "simulasi" in sumber.lower():
        perintah += ["--out", str(out / "metrics_sintetis.json")]

    env = {**os.environ, "PYTHONUTF8": "1"}
    hasil = subprocess.run(perintah, cwd=AKAR, env=env, capture_output=True, text=True)
    if hasil.returncode == 0:
        print(hasil.stdout.strip())
    else:
        print("[catatan] src.common.evaluate gagal, memakai penulisan metrik cadangan.")
        print("          penyebab:", (hasil.stderr or "").strip().splitlines()[-1])
        _, _, _, yva, _, _ = data.muat()
        tulis_metrik_cadangan(out / "metrics_sintetis.json", pred_val, yva, a.seed, durasi, a.perangkat, sumber)


if __name__ == "__main__":
    main()

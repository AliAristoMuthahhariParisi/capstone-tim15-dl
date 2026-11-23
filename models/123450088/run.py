"""Model jalur RNN — Tim 15, NIM 123450088.

Arsitektur:
    Jaringan berulang (LSTM) atas urutan 32 frame spektrogram, dengan kepala
    penaksir masker per frame (time-frequency mask) dan klasifikasi 10 perintah.

Pemakaian:
    python models/123450088/run.py --seed 1 [--epochs 60] [--final]
"""
from __future__ import annotations

import argparse
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
    """LSTM atas urutan frame, kepala masker time-frequency, lalu klasifikasi."""
    import torch
    import torch.nn as nn

    class RNNMaskerSpektrogram(nn.Module):
        def __init__(self, jumlah_kelas: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=40, hidden_size=64, num_layers=1, batch_first=True)
            self.masker = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
            self.kepala = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64, jumlah_kelas),
            )

        def forward(self, x):
            urut, _ = self.lstm(x)                 # [B, frame, hidden]
            urut = urut * self.masker(urut)        # masker per frame
            return self.kepala(urut.mean(dim=1))

    return RNNMaskerSpektrogram(n_kelas)


def latih_dan_prediksi(seed: int, final: bool, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Model jalur RNN Tim 15")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
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
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    # Metrik pada data simulasi ditulis terpisah agar tidak menimpa metrics.json resmi
    if "simulasi" in sumber.lower():
        perintah += ["--out", str(out / "metrics_sintetis.json")]

    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run(perintah, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()

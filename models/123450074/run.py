"""Kerangka model jalur CNN 2D — Tim 15, NIM 123450074.

Arsitektur:
    CNN 2D dengan blok penaksir masker spektrogram (time-frequency mask) dan
    ekstraksi ciri konvolusi 2D untuk klasifikasi 10 kelas perintah ucapan.

Pemakaian:
    python models/123450074/run.py --seed 1 [--epochs 30] [--final]
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
JALUR = "cnn"


def bangun_cnn(n_kelas: int):
    """Bangun model CNN 2D penaksir masker spektrogram dan klasifikasi perintah."""
    import torch
    import torch.nn as nn

    class CNNMaskerSpektrogram(nn.Module):
        def __init__(self, jumlah_kelas: int) -> None:
            super().__init__()
            # Blok penaksir masker time-frequency [B, 1, F_time, F_freq]
            self.penaksir_masker = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            # Blok ekstraksi ciri konvolusi 2D pada spektrogram termasker
            self.fitur = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.kepala = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(64, jumlah_kelas),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            masker = self.penaksir_masker(x)
            x_bersih = x * masker
            ciri = self.fitur(x_bersih)
            return self.kepala(ciri)

    return CNNMaskerSpektrogram(n_kelas)


def latih_dan_prediksi(
    seed: int,
    final: bool,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Muat data protokol tim, latih CNN 2D, dan kembalikan prediksi val (dan test)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    sumber = data.sumber_data()

    assert Xtr.ndim == 3, f"Masukan spektrogram harus 3D [N, frame, bin], ditemukan {Xtr.shape}"
    n_kelas = len(data.KALIMAT)

    def ke_tensor(X: np.ndarray) -> torch.Tensor:
        # Tambah dimensi kanal [N, 1, frame, bin] untuk Conv2d
        return torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)

    Xtr_t = ke_tensor(Xtr)
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64), dtype=torch.long)
    Xva_t = ke_tensor(Xva)

    model = bangun_cnn(n_kelas=n_kelas)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"sumber data : {sumber}")
    print(f"masukan     : latih {tuple(Xtr_t.shape)}, validasi {tuple(Xva_t.shape)}")
    print(f"model       : CNN 2D penaksir masker ({n_param:,} parameter, keluaran {n_kelas} kelas)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rugi_fn = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        urutan = rng.permutation(len(Xtr_t))
        total_rugi = 0.0
        for i in range(0, len(urutan), batch_size):
            idx = urutan[i : i + batch_size]
            opt.zero_grad()
            keluar = model(Xtr_t[idx])
            rugi = rugi_fn(keluar, ytr_t[idx])
            rugi.backward()
            opt.step()
            total_rugi += float(rugi.item()) * len(idx)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred_ep = model(Xva_t).argmax(dim=1).numpy()
                acc_va = float((pred_ep == yva).mean())
            print(f"  epoch {ep:2d}/{epochs}  loss={total_rugi / len(Xtr_t):.4f}  val_acc={acc_va:.4f}")

    model.eval()
    with torch.no_grad():
        logits_val = model(Xva_t)
        assert logits_val.shape == (len(Xva), n_kelas), f"Bentuk keluaran tidak sesuai: {logits_val.shape}"
        pred_val = logits_val.argmax(dim=1).numpy().astype(np.int64)
        pred_test = (
            model(ke_tensor(Xte)).argmax(dim=1).numpy().astype(np.int64)
            if final and Xte is not None
            else None
        )

    return pred_val, pred_test, sumber


def main() -> None:
    ap = argparse.ArgumentParser(description="Kerangka jalur CNN 2D Tim 15")
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

"""Kerangka model jalur Autoencoder (AE) — Tim 15, NIM 123450082.

Arsitektur:
    Small U-Net Autoencoder (CNN 2D). Mengompresi dan merekonstruksi 
    spektrogram magnitudo untuk penaksiran masker/skor anomali.

Pemakaian:
    python models/123450082/run.py --seed 1 [--epochs 30] [--final]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


def bangun_ae() -> nn.Module:
    """Bangun model Autoencoder 2D (Small U-Net) untuk rekonstruksi spektrogram."""
    class SmallAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
                # Asumsi spektrogram telah dinormalisasi 0-1
                nn.Sigmoid() 
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            terkompresi = self.encoder(x)
            rekonstruksi = self.decoder(terkompresi)
            return rekonstruksi

    return SmallAutoencoder()


def latih_dan_prediksi(
    seed: int,
    final: bool,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Muat data protokol tim, latih AE, dan kembalikan hasil rekonstruksi val (dan test)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    sumber = data.sumber_data()

    assert Xtr.ndim == 3, f"Masukan spektrogram harus 3D [N, frame, bin], ditemukan {Xtr.shape}"

    def ke_tensor(X: np.ndarray) -> torch.Tensor:
        # Tambah dimensi kanal [N, 1, frame, bin]
        return torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)

    Xtr_t = ke_tensor(Xtr)
    Xva_t = ke_tensor(Xva)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bangun_ae().to(device)
    Xtr_t = Xtr_t.to(device)
    Xva_t = Xva_t.to(device)

    n_param = sum(p.numel() for p in model.parameters())
    print(f"sumber data : {sumber}")
    print(f"masukan     : latih {tuple(Xtr_t.shape)}, validasi {tuple(Xva_t.shape)}")
    print(f"model       : Autoencoder (U-Net) ({n_param:,} parameter)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Autoencoder menggunakan MSE Loss karena merekonstruksi input
    rugi_fn = nn.MSELoss() 

    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        urutan = rng.permutation(len(Xtr_t))
        total_rugi = 0.0
        
        for i in range(0, len(urutan), batch_size):
            idx = urutan[i : i + batch_size]
            opt.zero_grad()
            keluar = model(Xtr_t[idx])
            # Target AE adalah input-nya sendiri
            rugi = rugi_fn(keluar, Xtr_t[idx]) 
            rugi.backward()
            opt.step()
            total_rugi += float(rugi.item()) * len(idx)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                keluar_va = model(Xva_t)
                rugi_va = float(rugi_fn(keluar_va, Xva_t).item())
            print(f"  epoch {ep:2d}/{epochs}  loss_train={total_rugi / len(Xtr_t):.4f}  loss_val={rugi_va:.4f}")

    # Eksport hasil prediksi (Untuk AE, kita export rekonstruksinya)
    model.eval()
    with torch.no_grad():
        # Luthia menggunakan astype(int64) karena dia klasifikasi, 
        # AE menggunakan float karena ini data gambar/spektrogram
        pred_val = model(Xva_t).cpu().numpy().astype(np.float32)
        
        pred_test = None
        if final and Xte is not None:
            Xte_t = ke_tensor(Xte).to(device)
            pred_test = model(Xte_t).cpu().numpy().astype(np.float32)

    return pred_val, pred_test, sumber


def main() -> None:
    ap = argparse.ArgumentParser(description="Kerangka jalur Autoencoder (AE) Tim 15")
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

    # Pemanggilan CLI evaluate yang sama persis dengan Luthia
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

    if "simulasi" in sumber.lower():
        perintah += ["--out", str(out / "metrics_sintetis.json")]

    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run(perintah, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()
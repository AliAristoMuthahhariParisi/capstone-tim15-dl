"""Kerangka model jalur Autoencoder (AE) — Tim 15, NIM 123450082.

Arsitektur:
    Small U-Net Autoencoder (CNN 2D). Mengompresi dan merekonstruksi 
    spektrogram magnitudo untuk penaksiran masker/skor anomali.

Pemakaian:
    python models/123450082/run.py --seed 1 [--epochs 30] [--final]
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# TAMBAHKAN PENAMBAL SYS.PATH 
AKAR = Path(__file__).resolve().parents[2]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

# LAKUKAN IMPORT SRC
from src.protokol.data import muat
from src.common import manifest, split

# TETAPKAN SEED
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

# DEFINISIKAN ARSITEKTUR AUTOENCODER (CNN)
class DenoisingCNNAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingCNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder():
    t_mulai = time.time()  # Catat waktu mulai
    
    print("Memuat data primer meja...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    
    X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    X_train_cnn = X_train_tensor.unsqueeze(1)
    X_val_tensor = torch.tensor(X_va, dtype=torch.float32)
    X_val_cnn = X_val_tensor.unsqueeze(1)

    model = DenoisingCNNAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 50 
    print(f"\nMemulai pelatihan Autoencoder dengan Seed {SEED}")
    
    model.train() 
    for epoch in range(epochs):
        optimizer.zero_grad() 
        output = model(X_train_cnn)
        loss = criterion(output, X_train_cnn)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss MSE: {loss.item():.4f}")

    print("\nMemulai evaluasi pada data validasi")
    model.eval() 
    
    with torch.no_grad():
        val_pred = model(X_val_cnn)
        
        def hitung_si_sdr(referensi, prediksi):
            eps = 1e-8
            alpha = np.sum(referensi * prediksi) / (np.sum(referensi**2) + eps)
            target_scaled = alpha * referensi
            noise = prediksi - target_scaled
            return 10 * np.log10((np.sum(target_scaled**2) + eps) / (np.sum(noise**2) + eps))

        si_sdr_list = []
        pred_numpy = val_pred.squeeze().numpy() 
        target_numpy = X_val_tensor.numpy()

        for i in range(len(pred_numpy)):
            ref = target_numpy[i].flatten()
            prd = pred_numpy[i].flatten()
            si_sdr_list.append(hitung_si_sdr(ref, prd))
            
        si_sdr_val = np.mean(si_sdr_list)
        print(f"Evaluasi Selesai! Rata-rata SI-SDR Validasi: {si_sdr_val:.4f} dB")

    durasi = time.time() - t_mulai  # Hitung total durasi

    # SIMPAN HASIL KE METRICS.JSON 
    hasil = {
        "metrik": {
            "si_sdr_peningkatan": round(float(si_sdr_val), 6),
            "akurasi_perintah": 0.0
        },
        "seed": SEED,
        "data_hash": manifest.data_hash(f"split:{split.SEED_SPLIT}".encode()),
        "perangkat": "cpu laptop",
        "model": "ae",
        "n_val": int(len(y_va)),
        "n_test": 0,
        "durasi_detik": round(durasi, 1)
    }
    
    output_path = Path("results/123450082/metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
        f.write("\n")
        
    print(f"\nSelesai! Hasil metrik validasi berhasil dicatat di: {output_path}")

if __name__ == "__main__":
    train_autoencoder()
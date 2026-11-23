"""Kerangka model jalur Autoencoder (AE) — Tim 15, NIM 123450082.

Arsitektur:
    Small U-Net Autoencoder (CNN 2D). Mengompresi dan merekonstruksi 
    spektrogram magnitudo untuk penaksiran masker/skor anomali.

Pemakaian:
    python models/123450082/run.py --seed 1 [--epochs 30] [--final]
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np

# Impor protokol tim
from src.protokol.data import muat
from src.common.manifest import data_hash

# 1. TETAPKAN SEED (Syarat Kartu #16)
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

# 2. DEFINISIKAN ARSITEKTUR AUTOENCODER (CNN)
class DenoisingCNNAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingCNNAutoencoder, self).__init__()
        
        # ENCODER: Memperkecil dimensi dari 32x40
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # DECODER: Memperbesar kembali ke 32x40
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder():
    # 3. MUAT DATA PRIMER
    print("Memuat data primer meja...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    
    # Ubah ke tensor PyTorch dan sesuaikan dimensi untuk CNN (tambah dimensi Channel di index 1)
    X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    X_train_cnn = X_train_tensor.unsqueeze(1)

    X_val_tensor = torch.tensor(X_va, dtype=torch.float32)
    X_val_cnn = X_val_tensor.unsqueeze(1)

    # Inisialisasi Model
    model = DenoisingCNNAutoencoder()

    # 4. PELATIHAN MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 50 
    print(f"\nMemulai pelatihan Autoencoder dengan Seed {SEED}...")
    
    model.train() 
    for epoch in range(epochs):
        optimizer.zero_grad() 
        
        output = model(X_train_cnn)
        
        # PERBAIKAN: Membandingkan output model dengan inputnya sendiri (X_train_cnn)
        loss = criterion(output, X_train_cnn)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss MSE: {loss.item():.4f}")

    # 5. EVALUASI PADA DATA VALIDASI
    print("\nMemulai evaluasi pada data validasi...")
    model.eval() 
    
    with torch.no_grad():
        val_pred = model(X_val_cnn)
        
        # Fungsi matematis SI-SDR
        def hitung_si_sdr(referensi, prediksi):
            eps = 1e-8
            alpha = np.sum(referensi * prediksi) / (np.sum(referensi**2) + eps)
            target_scaled = alpha * referensi
            noise = prediksi - target_scaled
            return 10 * np.log10((np.sum(target_scaled**2) + eps) / (np.sum(noise**2) + eps))

        si_sdr_list = []
        pred_numpy = val_pred.squeeze().numpy() 
        
        # PERBAIKAN: Menggunakan X_val_tensor.numpy()
        target_numpy = X_val_tensor.numpy()

        for i in range(len(pred_numpy)):
            ref = target_numpy[i].flatten()
            prd = pred_numpy[i].flatten()
            si_sdr_list.append(hitung_si_sdr(ref, prd))
            
        si_sdr_val = np.mean(si_sdr_list)
        print(f"Evaluasi Selesai! Rata-rata SI-SDR Validasi: {si_sdr_val:.4f} dB")

    # 6. SIMPAN HASIL KE METRICS.JSON
    output_dir = "results/123450082"
    os.makedirs(output_dir, exist_ok=True)
    
    hasil = {
        "model": "Autoencoder CNN (Sprint 1)",
        "seed": SEED,
        "data_hash": data_hash(),
        "metrics": {
            "si_sdr_peningkatan_val": round(float(si_sdr_val), 4),
            "akurasi_perintah_val": 0.0
        }
    }
    
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(hasil, f, indent=4)
        
    print(f"\nSelesai! Hasil metrik validasi berhasil dicatat di: {output_path}")

if __name__ == "__main__":
    train_autoencoder()
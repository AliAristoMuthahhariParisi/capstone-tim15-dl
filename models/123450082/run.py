import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Mengimpor pemuat data dan fungsi evaluasi dari repositori tim
from src.protokol.data import muat
from src.common import evaluate

# 2. Definisi Arsitektur Model Autoencoder (Small U-Net / CNN AE)
class SmallAutoencoder(nn.Module):
    def __init__(self):
        super(SmallAutoencoder, self).__init__()
        # Bagian Encoder (Mengompresi Spektrogram)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bagian Decoder (Merekonstruksi Spektrogram Kembali)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            # Sigmoid digunakan dengan asumsi nilai magnitudo dinormalisasi 0 - 1
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x adalah input spektrogram dengan shape (batch, channel=1, freq, time)
        fitur_terkompresi = self.encoder(x)
        rekonstruksi = self.decoder(fitur_terkompresi)
        return rekonstruksi

def main():
    # 3. Parsing Argumen Terminal (--seed)
    parser = argparse.ArgumentParser(description="Pelatihan Model Autoencoder")
    parser.add_argument("--seed", type=int, required=True, help="Seed untuk inisialisasi acak")
    args = parser.parse_args()

    # Menerapkan seed agar hasilnya bisa direproduksi
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[*] Menjalankan model AE dengan seed: {args.seed}")

    # Menentukan perangkat keras (GPU jika ada, jika tidak CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Menggunakan perangkat: {device}")

    # 4. Memuat Data (Fungsi muat() dari Integrator/Penjaga Data)
    print("[*] Memuat data simulasi...")
    # Asumsi fungsi muat() mengembalikan loader latih dan uji
    train_loader, test_loader = muat() 

    # 5. Inisialisasi Model, Loss, dan Optimizer
    model = SmallAutoencoder().to(device)
    # Autoencoder bertugas membuat output semirip mungkin dengan input,
    # jadi kita menggunakan MSE (Mean Squared Error) Loss
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6. Proses Pelatihan (Epoch singkat untuk simulasi)
    print("[*] Memulai pelatihan...")
    epochs = 2 
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            
            # Pada Autoencoder, target rekonstruksinya adalah inputnya itu sendiri
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"    Epoch {epoch+1}/{epochs} selesai | Rata-rata Loss: {total_loss/len(train_loader):.4f}")

    # 7. Evaluasi dan Penulisan metrics.json
    print("[*] Mengevaluasi model dan menyimpan hasil...")
    evaluate(model, test_loader, device)
    
    print("[*] Selesai! Pipeline berjalan dengan sukses.")

if __name__ == "__main__":
    main()
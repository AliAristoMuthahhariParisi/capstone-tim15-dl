import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.protokol import data as protokol_data

NIM = "122450021"
JADWAL_DEFAULT = BASE_DIR / "sesi-simulasi" / "tim-15-2026-11-12" / "jadwal.csv"
RESULTS_DIR = BASE_DIR / "results" / NIM


def muat_dataset(mode: str, jadwal_path: Path):
    """Memuat pasangan spektrogram/fitur (X) dan label kalimat (y) dari modul protokol."""
    if mode == "simulasi":
        if not jadwal_path.exists():
            raise FileNotFoundError(f"Berkas jadwal simulasi tidak ditemukan: {jadwal_path}")
        if hasattr(protokol_data, "muat_simulasi"):
            return protokol_data.muat_simulasi(jadwal_path)
        if hasattr(protokol_data, "load_simulasi"):
            return protokol_data.load_simulasi(jadwal_path)
        if hasattr(protokol_data, "load_data"):
            return protokol_data.load_data(jadwal_path=jadwal_path, mode="simulasi")
        raise AttributeError("Fungsi pemuat simulasi belum tersedia di src/protokol/data.py.")
    else:
        if hasattr(protokol_data, "load_data"):
            return protokol_data.load_data(mode="riil")
        raise NotImplementedError("Pemuat data riil belum diimplementasikan.")


def siapkan_spektrogram_3kanal(X: np.ndarray) -> np.ndarray:
    """
    Memastikan input spektrogram berbentuk 4D (N, 3, H, W) agar siap diproses
    oleh backbone pre-trained CNN untuk ekstraksi pola spasial masker bising.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        # Jika input masih vektor 1D per sampel, bentuk ulang ke matriks 2D mendekati persegi
        n, f = X.shape
        sisi = int(np.ceil(np.sqrt(f)))
        pad = sisi * sisi - f
        if pad > 0:
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
        X = X.reshape(n, sisi, sisi)

    if X.ndim == 3:
        # Replikasi 1 kanal spektrogram menjadi 3 kanal (N, 3, H, W)
        X = np.repeat(X[:, np.newaxis, :, :], 3, axis=1)
    elif X.ndim == 4 and X.shape[-1] in (1, 3):
        # Ubah (N, H, W, C) menjadi (N, C, H, W)
        X = np.transpose(X, (0, 3, 1, 2))
        if X.shape[1] == 1:
            X = np.repeat(X, 3, axis=1)

    return X


def ekstrak_fitur_pretrained(X: np.ndarray) -> np.ndarray:
    """
    Mengekstrak representasi spasial spektrogram menggunakan bobot MobileNetV2 pre-trained.
    """
    X_spec = siapkan_spektrogram_3kanal(X)

    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MobileNet_V2_Weights.DEFAULT
        backbone = mobilenet_v2(weights=weights).features.to(device)
        backbone.eval()

        fitur_list = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X_spec), batch_size):
                batch = torch.tensor(X_spec[i : i + batch_size], device=device)
                # Resize spasial spektrogram ke resolusi standar MobileNetV2
                batch_resized = F.interpolate(batch, size=(128, 128), mode="bilinear", align_corners=False)
                # Normalisasi per-sampel
                mean = batch_resized.mean(dim=(2, 3), keepdim=True)
                std = batch_resized.std(dim=(2, 3), keepdim=True) + 1e-6
                batch_norm = (batch_resized - mean) / std

                feat_map = backbone(batch_norm)
                # Global Average Pooling untuk merangkum pola spasial masker bising
                pooled = feat_map.mean(dim=(2, 3))
                fitur_list.append(pooled.cpu().numpy())

        return np.vstack(fitur_list)

    except Exception as e:
        print(f"[{NIM}] Peringatan PyTorch/Weights ({e}), menggunakan fallback perataan spasial.")
        return X_spec.reshape(len(X_spec), -1)


def main():
    parser = argparse.ArgumentParser(description=f"Transfer Learning Spektrogram - NIM {NIM}")
    parser.add_argument("--mode", choices=["simulasi", "riil"], default="simulasi")
    parser.add_argument("--jadwal", type=Path, default=JADWAL_DEFAULT)
    args = parser.parse_args()

    print(f"[{NIM}] Memuat data spektrogram (mode: {args.mode})...")
    X, y = muat_dataset(args.mode, args.jadwal)

    print(f"[{NIM}] Mengekstrak fitur spasial menggunakan backbone pre-trained...")
    X_fitur = ekstrak_fitur_pretrained(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_fitur, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    hasil = {
        "nim": NIM,
        "pendekatan": "Transfer Learning (MobileNetV2 Pre-trained Feature Extractor)",
        "mode": args.mode,
        "jumlah_sampel": int(len(y)),
        "dimensi_fitur_spasial": int(X_fitur.shape[1]),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / ("metrics_sintetis.json" if args.mode == "simulasi" else "metrics.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2)

    print(f"[{NIM}] Selesai! Accuracy: {hasil['accuracy']} | F1-Macro: {hasil['f1_macro']}")


if __name__ == "__main__":
    main()
import argparse
import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.protokol import data as protokol_data

NIM = "122450021"
RESULTS_DIR = BASE_DIR / "results" / NIM


def format_spektrogram_3kanal(X: np.ndarray) -> np.ndarray:
    """
    Mengubah spektrogram magnitudo 4 detik ke format 4D (N, 3, H, W)
    agar kompatibel dengan backbone CNN pralatih (3 kanal RGB).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        n, f = X.shape
        sisi = int(np.ceil(np.sqrt(f)))
        pad = sisi * sisi - f
        if pad > 0:
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
        X = X.reshape(n, sisi, sisi)

    if X.ndim == 3:
        # (N, F, T) -> (N, 3, F, T)
        X = np.repeat(X[:, np.newaxis, :, :], 3, axis=1)
    elif X.ndim == 4:
        if X.shape[-1] in (1, 3) and X.shape[1] not in (1, 3):
            X = np.transpose(X, (0, 3, 1, 2))
        if X.shape[1] == 1:
            X = np.repeat(X, 3, axis=1)
    return X


def latih_dan_prediksi_tl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """
    Adaptasi model pralatih (MobileNetV2) untuk menaksir pola spasial spektrogram magnitudo 4s.
    Impor torch dilakukan di dalam fungsi mengikuti konvensi starter tim.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_3c = format_spektrogram_3kanal(X_train)
    X_ev_3c = format_spektrogram_3kanal(X_eval)

    # Deteksi jenis tugas otomatis berdasarkan bentuk target y_train
    y_train_arr = np.asarray(y_train)
    is_klasifikasi = y_train_arr.ndim == 1 and np.issubdtype(y_train_arr.dtype, np.integer)

    if is_klasifikasi:
        kelas_unik = np.unique(y_train_arr)
        num_classes = int(kelas_unik.max()) + 1
        y_tr_tensor = torch.tensor(y_train_arr, dtype=torch.long)
    else:
        target_shape = y_train_arr.shape[1:]
        out_dim = int(np.prod(target_shape))
        y_tr_tensor = torch.tensor(y_train_arr.reshape(len(y_train_arr), -1), dtype=torch.float32)

    # Muat bobot pralatih MobileNetV2 (fallback ke weights=None jika offline tanpa cache)
    try:
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    except Exception:
        backbone = mobilenet_v2(weights=None).features

    # Bekukan blok awal agar ekstraksi fitur spasial efisien dan tidak overfitting
    for param in list(backbone.parameters())[:-15]:
        param.requires_grad = False

    in_features = 1280
    head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes if is_klasifikasi else out_dim),
    )

    backbone = backbone.to(device)
    head = head.to(device)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss() if is_klasifikasi else nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_3c), y_tr_tensor),
        batch_size=16,
        shuffle=True,
    )

    # Tahap Pelatihan / Adaptasi
    backbone.train()
    head.train()
    for _ in range(max(1, epochs)):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Resize spasial spektrogram magnitudo 4s ke resolusi kompatibel backbone
            bx_res = F.interpolate(bx, size=(128, 128), mode="bilinear", align_corners=False)
            feat = backbone(bx_res).mean(dim=(2, 3))
            logits = head(feat)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Tahap Inferensi Evaluasi
    backbone.eval()
    head.eval()
    preds = []
    eval_loader = DataLoader(TensorDataset(torch.tensor(X_ev_3c)), batch_size=32, shuffle=False)
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            bx_res = F.interpolate(bx, size=(128, 128), mode="bilinear", align_corners=False)
            feat = backbone(bx_res).mean(dim=(2, 3))
            out = head(feat)
            if is_klasifikasi:
                preds.append(out.argmax(dim=1).cpu().numpy())
            else:
                preds.append(out.view(len(bx), *target_shape).cpu().numpy())

    return np.concatenate(preds, axis=0)


def muat_data_protokol(final: bool):
    """Memanggil src.protokol.data.muat() sesuai tanda tangan fungsi yang tersedia."""
    sig = inspect.signature(protokol_data.muat)
    if "final" in sig.parameters:
        return protokol_data.muat(final=final)
    return protokol_data.muat()


def jalankan_evaluasi_cli(pred_path: Path, y_true_path: Path):
    """Memanggil src.common.evaluate lewat subprocess menyesuaikan argumen CLI yang tersedia."""
    help_out = subprocess.run(
        [sys.executable, "-m", "src.common.evaluate", "--help"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    ).stdout

    cmd = [sys.executable, "-m", "src.common.evaluate"]
    if "--nim" in help_out:
        cmd.extend(["--nim", NIM])
    if "--pred" in help_out:
        cmd.extend(["--pred", str(pred_path)])
    if "--true" in help_out:
        cmd.extend(["--true", str(y_true_path)])
    elif "--y-true" in help_out:
        cmd.extend(["--y-true", str(y_true_path)])

    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def main():
    parser = argparse.ArgumentParser(description=f"[TL] Transfer Learning Spektrogram 4s - NIM {NIM}")
    parser.add_argument("--seed", type=int, default=1, help="Seed acak untuk reproduksibilitas")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch adaptasi model")
    parser.add_argument("--final", action="store_true", help="Gunakan split pengujian final")
    args = parser.parse_args()

    print(f"[{NIM}] Memuat dataset lewat src.protokol.data.muat() (seed={args.seed}, final={args.final})...")
    data = muat_data_protokol(final=args.final)

    if isinstance(data, dict):
        X_train, y_train = data["X_train"], data["y_train"]
        X_eval = data["X_test"] if (args.final and "X_test" in data) else data["X_val"]
        y_eval = data["y_test"] if (args.final and "y_test" in data) else data["y_val"]
        is_simulasi = bool(data.get("is_simulasi", not args.final))
    else:
        X_train, y_train, X_eval, y_eval = data[0], data[1], data[2], data[3]
        is_simulasi = not args.final

    print(f"[{NIM}] Bentuk masukan spektrogram 4s: train={np.shape(X_train)}, eval={np.shape(X_eval)}")
    y_pred = latih_dan_prediksi_tl(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"[{NIM}] Bentuk keluaran prediksi: {y_pred.shape} (target: {np.shape(y_eval)})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / "pred_val.npy"
    y_true_path = RESULTS_DIR / "y_val.npy"
    np.save(pred_path, y_pred)
    np.save(y_true_path, y_eval)

    jalankan_evaluasi_cli(pred_path, y_true_path)

    # Aturan tim: metrik simulasi masuk ke metrics_sintetis.json, bukan metrics.json
    metrics_json = RESULTS_DIR / "metrics.json"
    metrics_sintetis = RESULTS_DIR / "metrics_sintetis.json"
    if is_simulasi and metrics_json.exists():
        shutil.move(str(metrics_json), str(metrics_sintetis))
        print(f"[{NIM}] Metrik simulasi disimpan ke {metrics_sintetis.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
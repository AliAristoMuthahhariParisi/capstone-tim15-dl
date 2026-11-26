import argparse
import inspect
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.protokol import data as protokol_data

NIM = "122450021"
RESULTS_DIR = BASE_DIR / "results" / NIM


def format_spektrogram_3kanal(X: np.ndarray) -> np.ndarray:
    """Mengubah spektrogram magnitudo ke format 4D (N, 3, H, W) untuk backbone CNN pralatih."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        n, f = X.shape
        sisi = int(np.ceil(np.sqrt(f)))
        pad = sisi * sisi - f
        if pad > 0:
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
        X = X.reshape(n, sisi, sisi)

    if X.ndim == 3:
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
    lr: float = 1e-3,
    freeze_mode: str = "partial",
) -> np.ndarray:
    """Adaptasi model pralatih MobileNetV2 pada masukan spektrogram magnitudo."""
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

    try:
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    except Exception:
        backbone = mobilenet_v2(weights=None).features

    params = list(backbone.parameters())
    if freeze_mode == "full":
        for param in params:
            param.requires_grad = False
    elif freeze_mode == "none":
        for param in params:
            param.requires_grad = True
    else:
        for param in params[:-15]:
            param.requires_grad = False

    head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes if is_klasifikasi else out_dim),
    )

    backbone = backbone.to(device)
    head = head.to(device)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(head.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss() if is_klasifikasi else nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_3c), y_tr_tensor),
        batch_size=16,
        shuffle=True,
    )

    backbone.train()
    head.train()
    for ep in range(1, max(1, epochs) + 1):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx_res = F.interpolate(bx, size=(128, 128), mode="bilinear", align_corners=False)
            feat = backbone(bx_res).mean(dim=(2, 3))
            logits = head(feat)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{NIM}] Epoch {ep}/{epochs} - Loss: {total_loss / max(len(train_loader), 1):.4f}")

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
    sig = inspect.signature(protokol_data.muat)
    if "final" in sig.parameters:
        return protokol_data.muat(final=final)
    return protokol_data.muat()


def jalankan_evaluasi_cli(pred_path: Path, seed: int, durasi: float):
    """Memanggil src.common.evaluate sesuai argumen wajib CLI."""
    import torch

    perangkat = "cuda" if torch.cuda.is_available() else "cpu laptop"
    cmd = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        "tl",
        "--seed",
        str(seed),
        "--pred-val",
        str(pred_path),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        perangkat,
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def main():
    parser = argparse.ArgumentParser(description=f"[TL] Transfer Learning Spektrogram - NIM {NIM}")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-mode", choices=["partial", "full", "none"], default="partial")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    t_mulai = time.time()
    data = muat_data_protokol(final=args.final)
    if isinstance(data, dict):
        X_train, y_train = data["X_train"], data["y_train"]
        X_eval = data["X_test"] if (args.final and "X_test" in data) else data["X_val"]
    else:
        X_train, y_train, X_eval = data[0], data[1], data[2]

    y_pred = latih_dan_prediksi_tl(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        freeze_mode=args.freeze_mode,
    )
    durasi = time.time() - t_mulai

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / "pred_val.npy"
    np.save(pred_path, y_pred)

    jalankan_evaluasi_cli(pred_path, seed=args.seed, durasi=durasi)

    y_val_sisa = RESULTS_DIR / "y_val.npy"
    if y_val_sisa.exists():
        y_val_sisa.unlink()

    metrics_json = RESULTS_DIR / "metrics.json"
    if metrics_json.exists():
        print(f"[{NIM}] Selesai! Hasil tersimpan di {metrics_json.relative_to(BASE_DIR)}")
        print(metrics_json.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

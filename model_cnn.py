"""
MODEL CNN (PyTorch) untuk klasifikasi pemicu deforestasi dari tumpukan citra 6 band.

Sesuai materi: "tumpukan semua band Sentinel-1 dan Sentinel-2 dipandang sebagai satu citra
multi-band". Pada CNN ini, 6 band masuk sebagai 6 channel input sekaligus. Model tidak diberi
tahu kapan harus mengandalkan radar atau optik; bobot konvolusi mempelajarinya sendiri
(mis. saat optik tertutup awan, ia belajar lebih banyak memakai radar).
"""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import data_sintetis as ds

torch.set_num_threads(max(1, torch.get_num_threads()))


def atur_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNNDriver(nn.Module):
    """CNN kecil: 4 blok konvolusi -> pooling global (rata-rata + maksimum) -> klasifikasi."""

    def __init__(self, n_band=6, n_kelas=len(ds.KELAS)):
        super().__init__()

        def blok(masuk, keluar, pool):
            lapis = [nn.Conv2d(masuk, keluar, 3, padding=1), nn.BatchNorm2d(keluar), nn.ReLU()]
            if pool:
                lapis.append(nn.MaxPool2d(2))
            return nn.Sequential(*lapis)

        self.fitur = nn.Sequential(
            blok(n_band, 16, True),   # 32x32 -> 16x16
            blok(16, 32, True),       # 16x16 -> 8x8
            blok(32, 64, True),       # 8x8   -> 4x4
            blok(64, 64, False),
        )
        self.kepala = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_kelas)
        )

    def forward(self, x):
        h = self.fitur(x)
        rata = h.mean(dim=(2, 3))
        maks = h.amax(dim=(2, 3))
        return self.kepala(torch.cat([rata, maks], dim=1))


def statistik_band(X):
    """Rata-rata dan simpangan baku per band dari data latih (untuk standardisasi)."""
    return X.mean(axis=(0, 2, 3)).astype(np.float32), X.std(axis=(0, 2, 3)).astype(np.float32)


def _ke_tensor(X, stat):
    mean, std = stat
    return torch.from_numpy((X - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-6))


def augmentasi(xb):
    """Flip dan rotasi 90 derajat acak. Aman untuk citra satelit (tidak ada 'atas' yang baku)."""
    if np.random.rand() < 0.5:
        xb = torch.flip(xb, dims=[3])
    if np.random.rand() < 0.5:
        xb = torch.flip(xb, dims=[2])
    return torch.rot90(xb, k=int(np.random.randint(4)), dims=[2, 3])


@torch.no_grad()
def prediksi_proba(model, stat, X, batch=256):
    model.eval()
    keluar = []
    for i in range(0, len(X), batch):
        logit = model(_ke_tensor(X[i:i + batch], stat))
        keluar.append(torch.softmax(logit, dim=1).numpy())
    return np.concatenate(keluar)


def latih(X_tr, y_tr, X_val, y_val, epoch=18, lr=3e-3, batch=64, seed=0, verbose=True):
    """Melatih CNN. Epoch terbaik dipilih berdasarkan macro-F1 pada data VALIDASI
    (bukan data uji), lalu bobot terbaik dipulihkan."""
    atur_seed(seed)
    stat = statistik_band(X_tr)
    Xt, yt = _ke_tensor(X_tr, stat), torch.from_numpy(y_tr)
    model = CNNDriver()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    langkah = int(np.ceil(len(Xt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epoch, steps_per_epoch=langkah)
    kriteria = nn.CrossEntropyLoss()

    terbaik, bobot_terbaik = -1.0, None
    riwayat = {"loss": [], "f1_val": [], "acc_val": []}
    t0 = time.time()
    for ep in range(1, epoch + 1):
        model.train()
        urutan = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = urutan[i:i + batch]
            xb, yb = augmentasi(Xt[idx]), yt[idx]
            opt.zero_grad()
            loss = kriteria(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        pred_val = prediksi_proba(model, stat, X_val).argmax(1)
        f1 = f1_score(y_val, pred_val, average="macro")
        acc = accuracy_score(y_val, pred_val)
        riwayat["loss"].append(total / len(Xt))
        riwayat["f1_val"].append(f1)
        riwayat["acc_val"].append(acc)
        if f1 > terbaik:
            terbaik = f1
            bobot_terbaik = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {ep:2d}/{epoch} | loss {total / len(Xt):.3f} | "
                  f"val acc {acc:.3f} | val macro-F1 {f1:.3f} | {time.time() - t0:5.0f} dtk")
    model.load_state_dict(bobot_terbaik)
    return model, stat, riwayat


def metrik(y_true, y_pred):
    """Ringkasan: accuracy, macro-F1, dan F1 per kelas."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_kelas": f1_score(y_true, y_pred, average=None,
                                 labels=range(len(ds.KELAS))).tolist(),
    }


def simpan_model(path, model, stat):
    torch.save({"state": model.state_dict(), "mean": stat[0], "std": stat[1]}, path)


def muat_model(path):
    ckpt = torch.load(path, weights_only=False)
    model = CNNDriver()
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, (ckpt["mean"], ckpt["std"])

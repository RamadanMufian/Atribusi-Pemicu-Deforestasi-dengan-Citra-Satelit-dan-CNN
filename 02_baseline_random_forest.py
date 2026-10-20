"""
LANGKAH 2 - BASELINE: MACHINE LEARNING KONVENSIONAL (Random Forest)

Di sesi tanya jawab, pemateri menyarankan: coba dulu metode konvensional sebagai tolok ukur
(benchmark), baru bandingkan apakah deep learning benar-benar meningkatkan akurasi dan layak
dari sisi waktu/komputasi. Skrip ini adalah tolok ukur tersebut.

Isi:
  1. Ekstraksi fitur manual (statistik band + bentuk area gangguan) -> tabel fitur.
  2. Random Forest, dilaporkan dengan accuracy DAN macro-F1.
  3. Ablasi sensor: hanya Sentinel-1, hanya Sentinel-2, atau gabungan keduanya.
     (menjawab pertanyaan peserta: kapan memakai S1 vs S2, dan apakah menggabungkan lebih baik?)

Hasil: output/baseline_rf.json (dipakai skrip CNN untuk perbandingan), output/rf_ablasi_sensor.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import data_sintetis as ds

FOLDER = Path(__file__).parent
OUT = FOLDER / "output"
OUT.mkdir(exist_ok=True)
SEED = 42


def fitur_patch(x, pakai_s1, pakai_s2):
    """Mengubah satu patch (6, 32, 32) menjadi vektor fitur klasik."""
    idx = (ds.IDX_S1 if pakai_s1 else []) + (ds.IDX_S2 if pakai_s2 else [])
    f = []
    for b in idx:  # statistik tiap band
        v = x[b].ravel()
        f += [v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 95)]

    if pakai_s2:  # indeks vegetasi NDVI dari NIR dan merah
        ndvi = (x[5] - x[4]) / (x[5] + x[4] + 1e-6)
        f += [ndvi.mean(), ndvi.std(), ndvi.min()]
    if pakai_s1:  # rasio polarisasi
        f += [(x[1] / (x[0] + 1e-6)).mean()]

    # Bentuk area gangguan: ambang batas pada band pendeteksi lalu hitung komponen terhubung
    if pakai_s2:
        mask = gaussian_filter(x[5], 1.0) < 0.47      # NIR turun = vegetasi hilang
    else:
        mask = gaussian_filter(x[1], 1.5) < 0.32      # VH turun = vegetasi hilang
    label_komp, n = label(mask)
    if n > 0:
        ukuran = np.bincount(label_komp.ravel())[1:]
        f += [mask.mean(), n, ukuran.max() / mask.size]
        ys, xs = np.nonzero(mask)
        if len(ys) > 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
            f += [np.sqrt(ev[1] / (ev[0] + 1e-3))]    # kelonjongan (jalan = sangat lonjong)
        else:
            f += [1.0]
    else:
        f += [0.0, 0, 0.0, 1.0]
    return f


def buat_tabel(X, pakai_s1, pakai_s2):
    return np.array([fitur_patch(x, pakai_s1, pakai_s2) for x in X], dtype=np.float32)


def main():
    print("=" * 62)
    print("LANGKAH 2 - BASELINE RANDOM FOREST")
    print("=" * 62)
    tr = ds.muat(FOLDER / "data" / "train.npz")
    te = ds.muat(FOLDER / "data" / "test.npz")

    skenario = {
        "Hanya S1 (radar)": (True, False),
        "Hanya S2 (optik)": (False, True),
        "Gabungan S1+S2": (True, True),
    }
    hasil = {}
    f1_per_kelas = {}
    for nama, (s1, s2) in skenario.items():
        t0 = time.time()
        Xtr, Xte = buat_tabel(tr["X"], s1, s2), buat_tabel(te["X"], s1, s2)
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(Xtr, tr["y"])
        pred = rf.predict(Xte)
        hasil[nama] = {
            "accuracy": float(accuracy_score(te["y"], pred)),
            "macro_f1": float(f1_score(te["y"], pred, average="macro")),
            "jumlah_fitur": int(Xtr.shape[1]),
            "waktu_detik": round(time.time() - t0, 1),
        }
        f1_per_kelas[nama] = f1_score(te["y"], pred, average=None,
                                      labels=range(len(ds.KELAS)))
        print(f"\n{nama}: {Xtr.shape[1]} fitur")
        print(f"  accuracy = {hasil[nama]['accuracy']:.3f} | macro-F1 = {hasil[nama]['macro_f1']:.3f}"
              f" | waktu = {hasil[nama]['waktu_detik']} detik")

    print("\nF1 per kelas (data uji):")
    print(f"  {'kelas':24s}" + "".join(f"{n[:16]:>18s}" for n in skenario))
    for k, nama_k in enumerate(ds.KELAS):
        print(f"  {nama_k:24s}" + "".join(f"{f1_per_kelas[n][k]:18.3f}" for n in skenario))

    print("\nCatatan: perhatikan kelas gangguan_alami (banjir) dan kebakaran: bandingkan F1-nya\n"
          "antara S1 saja dan S2 saja. Pemateri menyebut radar sensitif terhadap air dan kelembapan.")

    # simpan hasil gabungan untuk perbandingan dengan CNN
    gabung = "Gabungan S1+S2"
    with open(OUT / "baseline_rf.json", "w") as f:
        json.dump({**hasil[gabung], "f1_per_kelas": f1_per_kelas[gabung].tolist()}, f, indent=2)

    # grafik ablasi
    plt.figure(figsize=(9, 4))
    lebar = 0.27
    for i, nama in enumerate(skenario):
        plt.bar(np.arange(len(ds.KELAS)) + (i - 1) * lebar, f1_per_kelas[nama],
                lebar, label=nama)
    plt.xticks(range(len(ds.KELAS)), [k.replace("_", "\n") for k in ds.KELAS], fontsize=7)
    plt.ylabel("F1-score (data uji)")
    plt.title("Ablasi sensor dengan Random Forest")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "rf_ablasi_sensor.png", dpi=130)
    plt.close()
    print("\nTersimpan: output/baseline_rf.json, output/rf_ablasi_sensor.png")


if __name__ == "__main__":
    main()

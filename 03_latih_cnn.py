"""
LANGKAH 3 - MELATIH CNN (Deep Learning) UNTUK ATRIBUSI PEMICU

Alur (mengikuti workflow ML dan materi):
  tumpukan 6 band (2 radar + 4 optik) -> CNN -> kelas pemicu + probabilitas (tingkat keyakinan)

Yang dilaporkan:
  - accuracy DAN macro-F1. Pemateri menekankan bahwa accuracy keseluruhan didominasi kelas
    pertanian (kelas terbanyak), sehingga macro-F1 (tiap kelas berbobot sama) memberi gambaran
    yang lebih jujur.
  - F1 per kelas dan confusion matrix.
  - perbandingan dengan baseline Random Forest (langkah 2).

Hasil: output/cnn_driver.pt, output/cnn_test_proba.npy, output/cnn_hasil.json,
       output/cnn_kurva_latih.png, output/cnn_confusion_matrix.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import data_sintetis as ds
import model_cnn as mc

FOLDER = Path(__file__).parent
OUT = FOLDER / "output"
OUT.mkdir(exist_ok=True)


def main():
    print("=" * 62)
    print("LANGKAH 3 - CNN PEMICU DEFORESTASI (PyTorch)")
    print("=" * 62)
    tr = ds.muat(FOLDER / "data" / "train.npz")
    va = ds.muat(FOLDER / "data" / "val.npz")
    te = ds.muat(FOLDER / "data" / "test.npz")
    print(f"Train {len(tr['y'])} | Val {len(va['y'])} | Test {len(te['y'])} patch")

    print("\nMelatih CNN (epoch terbaik dipilih dari macro-F1 data validasi):")
    t0 = time.time()
    model, stat, riwayat = mc.latih(tr["X"], tr["y"], va["X"], va["y"], epoch=18, seed=0)
    waktu = time.time() - t0
    print(f"Waktu pelatihan: {waktu:.0f} detik pada CPU")

    proba = mc.prediksi_proba(model, stat, te["X"])
    pred = proba.argmax(1)
    hasil = mc.metrik(te["y"], pred)

    print("\n--- Evaluasi pada DATA UJI (tidak pernah dipakai saat pelatihan) ---")
    print(f"accuracy = {hasil['accuracy']:.3f} | macro-F1 = {hasil['macro_f1']:.3f}")
    selisih = hasil["accuracy"] - hasil["macro_f1"]
    print(f"Selisih accuracy - macro-F1 = {selisih:.3f}")
    if selisih > 0.03:
        print("  Accuracy tampak lebih baik daripada macro-F1 karena kelas besar (pertanian) "
              "mendominasi; kelas kecil lebih sulit.")
    else:
        print("  Selisihnya kecil di data sintetis ini. Pada data nyata dengan kelas yang jauh lebih "
              "tidak seimbang, selisih bisa membesar (inilah alasan pemateri melaporkan macro-F1).")

    print("\nF1 per kelas (CNN vs Random Forest gabungan S1+S2):")
    rf = None
    if (OUT / "baseline_rf.json").exists():
        rf = json.load(open(OUT / "baseline_rf.json"))
    print(f"  {'kelas':24s}{'CNN':>8s}{'RF':>8s}{'jumlah uji':>12s}")
    for k, nama in enumerate(ds.KELAS):
        rf_f1 = f"{rf['f1_per_kelas'][k]:8.3f}" if rf else f"{'-':>8s}"
        print(f"  {nama:24s}{hasil['f1_per_kelas'][k]:8.3f}{rf_f1}{int((te['y'] == k).sum()):12d}")

    if rf:
        print("\n--- Benchmark konvensional vs deep learning ---")
        print(f"  Random Forest : accuracy {rf['accuracy']:.3f} | macro-F1 {rf['macro_f1']:.3f} "
              f"| latih {rf['waktu_detik']} dtk (+ rekayasa fitur manual)")
        print(f"  CNN           : accuracy {hasil['accuracy']:.3f} | macro-F1 {hasil['macro_f1']:.3f} "
              f"| latih {waktu:.0f} dtk (tanpa fitur manual)")
        print("  Pertimbangkan apakah selisih akurasi sepadan dengan tambahan waktu dan komputasi.")
        print("  Catatan jujur: satu kali percobaan pada data sintetis, dan fitur manual RF di sini sederhana;\n"
              "  fitur yang lebih baik dapat mempersempit selisih. Jangan digeneralisasi ke data nyata.")

    # simpan artefak
    mc.simpan_model(OUT / "cnn_driver.pt", model, stat)
    np.save(OUT / "cnn_test_proba.npy", proba)
    with open(OUT / "cnn_hasil.json", "w") as f:
        json.dump({**hasil, "waktu_detik": round(waktu, 1)}, f, indent=2)

    # kurva pelatihan
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(riwayat["loss"])
    ax[0].set_title("Loss pelatihan")
    ax[0].set_xlabel("epoch")
    ax[1].plot(riwayat["acc_val"], label="accuracy val")
    ax[1].plot(riwayat["f1_val"], label="macro-F1 val")
    ax[1].set_title("Kinerja pada data validasi")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "cnn_kurva_latih.png", dpi=130)
    plt.close()

    # confusion matrix
    cm = confusion_matrix(te["y"], pred, labels=range(len(ds.KELAS)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=[k.replace("_", "\n") for k in ds.KELAS]).plot(
        ax=ax, cmap="Greens", colorbar=False, values_format=".2f", xticks_rotation=45)
    for t in ax.texts:
        t.set_fontsize(7)
    ax.tick_params(labelsize=7)
    plt.title("Confusion matrix CNN (dinormalkan per kelas aktual)")
    plt.tight_layout()
    plt.savefig(OUT / "cnn_confusion_matrix.png", dpi=130)
    plt.close()

    print("\nTersimpan di folder output/: cnn_driver.pt, cnn_test_proba.npy, cnn_hasil.json, grafik")


if __name__ == "__main__":
    main()

"""
LANGKAH 1 - MEMBUAT DATASET (Data Collection)

Membuat dataset sintetis dan membaginya menjadi train / validation / test.
Juga menampilkan:
  - contoh satu patch per kelas (komposit optik dan radar),
  - distribusi kelas (ketidakseimbangan kelas: pertanian mendominasi, seperti di materi).

Hasil: data/train.npz, data/val.npz, data/test.npz, output/contoh_patch.png,
       output/distribusi_kelas.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data_sintetis as ds

FOLDER = Path(__file__).parent
DATA = FOLDER / "data"
OUT = FOLDER / "output"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

N_TRAIN, N_VAL, N_TEST = 3000, 600, 1200


def komposit_optik(x):
    """Gambar warna alami dari band merah-hijau-biru Sentinel-2 (diregangkan agar jelas)."""
    rgb = np.stack([x[4], x[3], x[2]], axis=-1)
    return np.clip((rgb - 0.05) / 0.35, 0, 1)


def komposit_radar(x):
    """Gambar abu-abu dari VV radar Sentinel-1."""
    return np.clip(x[0] / 0.8, 0, 1)


def main():
    print("Membuat dataset sintetis ...")
    paket = {
        "train": ds.buat_dataset(N_TRAIN, seed=1),
        "val": ds.buat_dataset(N_VAL, seed=2),
        "test": ds.buat_dataset(N_TEST, seed=3),
    }
    for nama, d in paket.items():
        ds.simpan(DATA / f"{nama}.npz", d)
        print(f"  {nama:5s}: {len(d['y']):5d} patch, bentuk {d['X'].shape[1:]} (band, tinggi, lebar)")

    print("\nBand per patch:", ", ".join(ds.BAND))

    print("\nDistribusi kelas pada data latih:")
    jumlah = np.bincount(paket["train"]["y"], minlength=len(ds.KELAS))
    for k, j in zip(ds.KELAS, jumlah):
        print(f"  {k:24s} {j:5d}  ({j / jumlah.sum():5.1%})")
    pertanian = (jumlah[0] + jumlah[1]) / jumlah.sum()
    print(f"  -> Pertanian (kecil + besar) = {pertanian:.1%} dari seluruh data. "
          "Kelas ini akan mendominasi akurasi keseluruhan.")

    print("\nKomposisi kelas per wilayah (data latih):")
    d = paket["train"]
    for w, nama_w in enumerate(ds.WILAYAH):
        sub = d["y"][d["wilayah"] == w]
        top = np.argsort(np.bincount(sub, minlength=len(ds.KELAS)))[::-1][:2]
        ringkas = ", ".join(f"{ds.KELAS[t]} {np.mean(sub == t):.0%}" for t in top)
        print(f"  {nama_w:16s}: {ringkas}")

    # ----- gambar contoh: satu patch per kelas
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(2, len(ds.KELAS), figsize=(2.1 * len(ds.KELAS), 4.6))
    for k in range(len(ds.KELAS)):
        x, _ = ds.buat_patch(k, "Asia Tenggara", rng)
        ax[0, k].imshow(komposit_optik(x))
        ax[1, k].imshow(komposit_radar(x), cmap="gray", vmin=0, vmax=1)
        ax[0, k].set_title(ds.KELAS[k].replace("_", "\n"), fontsize=8)
        for a in ax[:, k]:
            a.axis("off")
    ax[0, 0].text(-0.08, 0.5, "Optik (S2)", transform=ax[0, 0].transAxes,
                  rotation=90, va="center", ha="right", fontsize=9)
    ax[1, 0].text(-0.08, 0.5, "Radar (S1 VV)", transform=ax[1, 0].transAxes,
                  rotation=90, va="center", ha="right", fontsize=9)
    plt.suptitle("Contoh patch sintetis per kelas pemicu (bukan citra satelit asli)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "contoh_patch.png", dpi=130)
    plt.close()

    # ----- gambar distribusi kelas
    plt.figure(figsize=(8, 3.6))
    plt.bar(range(len(ds.KELAS)), jumlah, color="#4C8C4A")
    plt.xticks(range(len(ds.KELAS)), [k.replace("_", "\n") for k in ds.KELAS], fontsize=7)
    plt.ylabel("Jumlah patch (data latih)")
    plt.title("Distribusi kelas: tidak seimbang")
    plt.tight_layout()
    plt.savefig(OUT / "distribusi_kelas.png", dpi=130)
    plt.close()

    print("\nTersimpan: data/*.npz, output/contoh_patch.png, output/distribusi_kelas.png")


if __name__ == "__main__":
    main()

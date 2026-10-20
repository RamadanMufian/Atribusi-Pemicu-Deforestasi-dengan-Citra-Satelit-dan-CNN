"""
LANGKAH 5 - ANALISIS MUSIMAN PEMICU

Materi: setelah peringatan dijalankan bulan demi bulan selama beberapa tahun, pemateri
menganalisis musiman tiap pemicu. Nilai indeks tinggi = gangguan terkonsentrasi di waktu
tertentu dalam setahun; nilai rendah = tersebar merata sepanjang tahun. Kebakaran dan
pembukaan lahan pertanian paling musiman; musiman lebih kuat di Amazon dan Lembah Kongo
daripada di Asia Tenggara.

Di sini kita meniru alurnya:
  1. membuat "aliran peringatan" baru (data sintetis, seolah hasil beberapa tahun),
  2. mengklasifikasikannya dengan CNN dari langkah 3,
  3. menghitung indeks musiman per pemicu dan wilayah dari HASIL PREDIKSI,
  4. membandingkannya dengan indeks dari label sebenarnya (tersedia hanya pada data sintetis)
     untuk melihat seberapa besar kesalahan klasifikasi memengaruhi analisis.

Indeks musiman yang dipakai (definisi buatan proyek ini, bukan definisi resmi pemateri):
    indeks = 1 - H / ln(12),  H = entropi sebaran bulanan
    0 = sama rata sepanjang tahun, 1 = seluruhnya terjadi di satu bulan.
H dikoreksi bias sampel kecil (koreksi Miller-Madow); sel dengan data terlalu sedikit dilewati.

Prasyarat: langkah 3 selesai (output/cnn_driver.pt).
Hasil: output/musiman_indeks.png, output/musiman_kebakaran.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data_sintetis as ds
import model_cnn as mc

FOLDER = Path(__file__).parent
OUT = FOLDER / "output"
MIN_DATA = 30
N_ALIRAN = 5000


def indeks_musiman(bulan):
    """Indeks musiman 0..1 dari daftar bulan kejadian (1..12)."""
    n = len(bulan)
    cacah = np.bincount(bulan, minlength=13)[1:].astype(float)
    p = cacah / n
    h = -np.sum(p[p > 0] * np.log(p[p > 0]))
    h += (np.count_nonzero(cacah) - 1) / (2 * n)          # koreksi Miller-Madow
    return float(np.clip(1 - h / np.log(12), 0, 1))


def tabel_indeks(kelas, wilayah, bulan):
    t = np.full((len(ds.KELAS), len(ds.WILAYAH)), np.nan)
    for k in range(len(ds.KELAS)):
        for w in range(len(ds.WILAYAH)):
            m = (kelas == k) & (wilayah == w)
            if m.sum() >= MIN_DATA:
                t[k, w] = indeks_musiman(bulan[m])
    return t


def main():
    print("=" * 62)
    print("LANGKAH 5 - ANALISIS MUSIMAN PEMICU")
    print("=" * 62)
    model, stat = mc.muat_model(OUT / "cnn_driver.pt")

    print(f"Membuat aliran peringatan sintetis ({N_ALIRAN} peringatan) dan mengklasifikasikannya ...")
    aliran = ds.buat_dataset(N_ALIRAN, seed=21)
    pred = mc.prediksi_proba(model, stat, aliran["X"]).argmax(1)
    print(f"  Accuracy pada aliran ini: {np.mean(pred == aliran['y']):.3f}")

    ind_pred = tabel_indeks(pred, aliran["wilayah"], aliran["bulan"])
    ind_benar = tabel_indeks(aliran["y"], aliran["wilayah"], aliran["bulan"])

    print("\nIndeks musiman (dari PREDIKSI CNN | dari label sebenarnya):")
    print(f"  {'pemicu':24s}" + "".join(f"{w[:15]:>20s}" for w in ds.WILAYAH))
    for k, nama in enumerate(ds.KELAS):
        baris = ""
        for w in range(len(ds.WILAYAH)):
            a, b = ind_pred[k, w], ind_benar[k, w]
            baris += f"{'n/a':>20s}" if np.isnan(a) else f"{a:>11.2f} |{b:>6.2f}   "
        print(f"  {nama:24s}{baris}")
    print("  (n/a = data kurang dari "
          f"{MIN_DATA} peringatan, indeks tidak dapat diandalkan)")

    # Ringkasan pola
    rata_pred_kelas = np.nanmean(ind_pred, axis=1)
    tertinggi = ds.KELAS[int(np.nanargmax(rata_pred_kelas))]
    rata_wilayah = np.nanmean(ind_pred, axis=0)
    print(f"\nPemicu paling musiman (rata-rata lintas wilayah, dari prediksi): {tertinggi}")
    print("Rata-rata indeks per wilayah: " + ", ".join(
        f"{w} {v:.2f}" for w, v in zip(ds.WILAYAH, rata_wilayah)))
    print("Nilai mutlak indeks tampak kecil karena indeks berbasis entropi 12 bulan bersifat\n"
          "konservatif; yang penting adalah PERBANDINGAN relatif antar pemicu dan antar wilayah.")
    print("Catatan: pola musiman ini sengaja ditanam di generator data (kebakaran musiman kuat, Asia\n"
          "Tenggara lemah) agar analisisnya bisa dipraktikkan. Pada data nyata, pola harus ditemukan\n"
          "dari data, dan klasifikasi yang keliru dapat menggeser indeks.")

    # Heatmap indeks
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(ind_pred, cmap="YlOrRd", vmin=0, vmax=max(0.05, np.nanmax(ind_pred)))
    ax.set_xticks(range(3), ds.WILAYAH, fontsize=8)
    ax.set_yticks(range(len(ds.KELAS)), [k.replace("_", " ") for k in ds.KELAS], fontsize=8)
    for k in range(len(ds.KELAS)):
        for w in range(3):
            if not np.isnan(ind_pred[k, w]):
                ax.text(w, k, f"{ind_pred[k, w]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, label="indeks musiman (tinggi = terkonsentrasi)")
    plt.title("Indeks musiman pemicu per wilayah (dari prediksi CNN)")
    plt.tight_layout()
    plt.savefig(OUT / "musiman_indeks.png", dpi=130)
    plt.close()

    # Profil bulanan kebakaran per wilayah
    plt.figure(figsize=(7, 3.8))
    for w, nama_w in enumerate(ds.WILAYAH):
        m = (pred == 5) & (aliran["wilayah"] == w)
        if m.sum() >= MIN_DATA:
            cacah = np.bincount(aliran["bulan"][m], minlength=13)[1:]
            plt.plot(range(1, 13), cacah / cacah.sum(), "o-", label=f"{nama_w} (n={int(m.sum())})")
    plt.axhline(1 / 12, color="gray", ls="--", lw=0.8, label="sebaran merata")
    plt.xlabel("Bulan")
    plt.ylabel("Proporsi peringatan kebakaran")
    plt.title("Profil bulanan kebakaran (prediksi CNN)")
    plt.xticks(range(1, 13))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "musiman_kebakaran.png", dpi=130)
    plt.close()

    print("\nTersimpan: output/musiman_indeks.png, output/musiman_kebakaran.png")


if __name__ == "__main__":
    main()

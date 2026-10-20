"""
LANGKAH 6 - INFERENCE BULANAN (Deployment sederhana)

Meniru rilis bulanan di materi: tiap bulan datang peringatan deforestasi baru; model
mengklasifikasikan pemicunya dan memberi label "tidak yakin" bila probabilitasnya rendah.

Keluaran: output/peringatan_bulanan.csv berisi tiap peringatan, pemicu prediksi, tingkat
keyakinan, dan status (diklasifikasi / perlu ditinjau manusia).

Catatan penting (dari materi):
  - Klasifikasi ini PERKIRAAN dan selalu bisa salah. Untuk penegakan hukum atau intervensi,
    hasilnya harus diverifikasi lapangan dan ditinjau manusia.
  - Data hanya diperbarui sebulan sekali, jadi ada latensi.
  - Yang diklasifikasi hanya pemicu LANGSUNG yang terlihat setelah gangguan, bukan penyebab
    sebenarnya yang lebih kompleks.

Prasyarat: langkah 3 selesai.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data_sintetis as ds
import model_cnn as mc

FOLDER = Path(__file__).parent
OUT = FOLDER / "output"
AMBANG_YAKIN = 0.70
N_BATCH = 400


def klasifikasikan(model, stat, patch, ambang=AMBANG_YAKIN):
    """Klasifikasi SATU patch (6, 32, 32). Mengembalikan (pemicu, probabilitas)."""
    p = mc.prediksi_proba(model, stat, patch[None])[0]
    k = int(p.argmax())
    return (ds.KELAS[k] if p[k] >= ambang else "tidak_yakin"), float(p[k])


def main():
    print("=" * 62)
    print("LANGKAH 6 - INFERENCE BULANAN")
    print("=" * 62)
    model, stat = mc.muat_model(OUT / "cnn_driver.pt")

    print(f"\nContoh klasifikasi satu peringatan:")
    contoh = ds.buat_dataset(1, seed=99)
    pemicu, prob = klasifikasikan(model, stat, contoh["X"][0])
    print(f"  -> {pemicu} (probabilitas {prob:.2f}); label sintetis sebenarnya: "
          f"{ds.KELAS[contoh['y'][0]]}")

    print(f"\nMengklasifikasikan batch bulan ini ({N_BATCH} peringatan baru) ...")
    batch = ds.buat_dataset(N_BATCH, seed=31)
    proba = mc.prediksi_proba(model, stat, batch["X"])
    kelas = proba.argmax(1)
    yakin = proba.max(1)

    df = pd.DataFrame({
        "id_peringatan": [f"ALR-{i:04d}" for i in range(N_BATCH)],
        "wilayah": [ds.WILAYAH[w] for w in batch["wilayah"]],
        "bulan": batch["bulan"],
        "pemicu_prediksi": [ds.KELAS[k] if c >= AMBANG_YAKIN else "tidak_yakin"
                            for k, c in zip(kelas, yakin)],
        "keyakinan": yakin.round(3),
        "status": np.where(yakin >= AMBANG_YAKIN, "diklasifikasi", "perlu ditinjau manusia"),
        # Kolom di bawah hanya ada karena datanya sintetis; pada data nyata tidak tersedia
        "label_sintetis_sebenarnya": [ds.KELAS[k] for k in batch["y"]],
    })
    df.to_csv(OUT / "peringatan_bulanan.csv", index=False)

    n_tinjau = int((df["status"] != "diklasifikasi").sum())
    print(f"\nRingkasan batch (ambang keyakinan {AMBANG_YAKIN}):")
    print(f"  Diklasifikasi otomatis      : {N_BATCH - n_tinjau} ({1 - n_tinjau / N_BATCH:.1%})")
    print(f"  Perlu ditinjau manusia      : {n_tinjau} ({n_tinjau / N_BATCH:.1%})")
    mask = df["status"] == "diklasifikasi"
    acc = np.mean(df.loc[mask, "pemicu_prediksi"] == df.loc[mask, "label_sintetis_sebenarnya"])
    print(f"  Accuracy yang diklasifikasi : {acc:.3f} (hanya dapat dihitung karena data sintetis)")

    print("\nJumlah peringatan per pemicu prediksi dan wilayah:")
    tabel = pd.crosstab(df["pemicu_prediksi"], df["wilayah"])
    print(tabel.to_string())

    print("\nContoh isi CSV (5 baris pertama):")
    print(df.head().to_string(index=False))
    print("\nIngat: hasil ini perkiraan model. Verifikasi lapangan tetap diperlukan sebelum\n"
          "dipakai untuk tindakan penegakan hukum atau keputusan penting.")
    print("\nTersimpan: output/peringatan_bulanan.csv")


if __name__ == "__main__":
    main()

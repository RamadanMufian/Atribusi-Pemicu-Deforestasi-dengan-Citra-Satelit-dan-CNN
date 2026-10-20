"""
LANGKAH 4 - EVALUASI LANJUTAN (mengikuti sesi tanya jawab di materi)

  A. Akurasi per wilayah        : materi menilai akurasi terpisah per wilayah.
  B. Kelas "tidak yakin"        : model dapat memberi label tidak yakin saat probabilitas rendah.
                                  Berapa cakupan yang hilang, dan berapa akurasi yang naik?
  C. Pengaruh tutupan awan      : optik terhambat awan, radar tidak. Bagaimana kinerja model?
  D. Kebakaran vs gangguan alami: pasangan kelas yang sulit dibedakan. Apa yang tertukar?
  E. Pelatihan ulang terarah    : menambah contoh kebakaran dan banjir (seperti di materi),
                                  dengan kelompok kontrol agar perbandingannya adil.

Prasyarat: jalankan langkah 1 dan 3 terlebih dahulu.
Hasil: output/evaluasi_lanjutan.json, output/kompromi_keyakinan.png, output/retrain_terarah.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import data_sintetis as ds
import model_cnn as mc

FOLDER = Path(__file__).parent
OUT = FOLDER / "output"
IDX_API, IDX_ALAMI = 5, 6


def garis(judul):
    print("\n" + "-" * 62 + f"\n{judul}\n" + "-" * 62)


def main():
    print("=" * 62)
    print("LANGKAH 4 - EVALUASI LANJUTAN")
    print("=" * 62)
    tr = ds.muat(FOLDER / "data" / "train.npz")
    va = ds.muat(FOLDER / "data" / "val.npz")
    te = ds.muat(FOLDER / "data" / "test.npz")
    proba = np.load(OUT / "cnn_test_proba.npy")
    pred, y = proba.argmax(1), te["y"]
    keyakinan = proba.max(1)
    hasil = {}

    # ------------------------------------------------------------------ A. per wilayah
    garis("A. Kinerja per wilayah")
    hasil["per_wilayah"] = {}
    for w, nama in enumerate(ds.WILAYAH):
        m = te["wilayah"] == w
        acc = accuracy_score(y[m], pred[m])
        mf1 = f1_score(y[m], pred[m], average="macro", labels=np.unique(y[m]), zero_division=0)
        hasil["per_wilayah"][nama] = {"n": int(m.sum()), "accuracy": acc, "macro_f1": mf1}
        print(f"  {nama:16s} n={int(m.sum()):4d} | accuracy {acc:.3f} | macro-F1 {mf1:.3f}")
    print("  Tujuannya: melihat apakah model bekerja lebih baik di wilayah tertentu.")

    # ------------------------------------------------------------------ B. tidak yakin
    garis('B. Label "tidak yakin" berdasarkan ambang probabilitas')
    print(f"  {'ambang':>7s}{'cakupan':>10s}{'accuracy':>11s}{'macro-F1':>11s}  (hanya prediksi yang yakin)")
    ambang_list = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    cakupan, akurasi = [], []
    hasil["ambang"] = {}
    for a in ambang_list:
        yakin = keyakinan >= a
        acc = accuracy_score(y[yakin], pred[yakin])
        mf1 = f1_score(y[yakin], pred[yakin], average="macro", labels=np.unique(y[yakin]),
                       zero_division=0)
        cakupan.append(yakin.mean())
        akurasi.append(acc)
        hasil["ambang"][str(a)] = {"cakupan": float(yakin.mean()), "accuracy": acc, "macro_f1": mf1}
        print(f"  {a:7.1f}{yakin.mean():10.1%}{acc:11.3f}{mf1:11.3f}")
    tdk = keyakinan < 0.7
    print("\n  Kelas yang paling sering berlabel 'tidak yakin' (ambang 0.7):")
    for k in np.argsort([(tdk & (y == k)).sum() / max((y == k).sum(), 1) for k in range(7)])[::-1][:3]:
        print(f"    {ds.KELAS[k]:24s} {(tdk & (y == k)).sum() / (y == k).sum():.1%} dari kasusnya")
    print("  Trade-off: ambang lebih tinggi -> lebih akurat, tetapi lebih banyak kasus perlu\n"
          "  ditinjau manusia (human oversight).")

    plt.figure(figsize=(5.5, 4))
    plt.plot(cakupan, akurasi, "o-")
    for a, c, k in zip(ambang_list, cakupan, akurasi):
        plt.annotate(f"{a:.1f}", (c, k), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Cakupan (proporsi prediksi yang diberi label)")
    plt.ylabel("Accuracy pada prediksi yang yakin")
    plt.title('Kompromi ambang keyakinan ("tidak yakin")')
    plt.tight_layout()
    plt.savefig(OUT / "kompromi_keyakinan.png", dpi=130)
    plt.close()

    # ------------------------------------------------------------------ C. awan
    garis("C. Pengaruh tutupan awan pada citra optik")
    kelompok = {
        "tanpa awan": te["awan"] == 0,
        "awan sebagian (<=50%)": (te["awan"] > 0) & (te["awan"] <= 0.5),
        "awan tebal (>50%)": te["awan"] > 0.5,
    }
    hasil["awan"] = {}
    for nama, m in kelompok.items():
        acc = accuracy_score(y[m], pred[m])
        hasil["awan"][nama] = {"n": int(m.sum()), "accuracy": acc}
        print(f"  {nama:24s} n={int(m.sum()):4d} | accuracy {acc:.3f}")
    print("  Saat optik tertutup awan, model hanya bisa mengandalkan radar (S1).")
    tebal = kelompok["awan tebal (>50%)"]
    f1_tebal = f1_score(y[tebal], pred[tebal], average=None, labels=range(7), zero_division=0)
    f1_semua = f1_score(y, pred, average=None, labels=range(7), zero_division=0)
    print("  F1 per kelas: seluruh data uji vs awan tebal (n kecil = angka kurang stabil):")
    for k in range(7):
        print(f"    {ds.KELAS[k]:24s} semua={f1_semua[k]:.3f} | awan tebal={f1_tebal[k]:.3f} "
              f"(selisih {f1_tebal[k] - f1_semua[k]:+.3f}, n={int((y[tebal] == k).sum())})")

    # ------------------------------------------------------------------ D. api vs alami
    garis("D. Pasangan sulit: kebakaran vs gangguan alami")
    cm = confusion_matrix(y, pred, labels=range(7), normalize="true")
    for k in (IDX_API, IDX_ALAMI):
        urut = np.argsort(cm[k])[::-1]
        ringkas = ", ".join(f"{ds.KELAS[j]} {cm[k, j]:.0%}" for j in urut[:3] if cm[k, j] > 0.005)
        print(f"  {ds.KELAS[k]:16s} diprediksi sebagai -> {ringkas}")
    print("  Materi menyebut kebakaran dan gangguan alami (banjir) sering tertukar, terutama di\n"
          "  dataran banjir tempat keduanya muncul berdampingan.")

    # ------------------------------------------------------------------ E. retraining
    garis("E. Pelatihan ulang terarah: menambah contoh kebakaran dan banjir")
    n_extra = 900
    extra_terarah = ds.buat_dataset(n_extra, seed=11, hanya_kelas=[IDX_API, IDX_ALAMI])
    extra_kontrol = ds.buat_dataset(n_extra, seed=12)  # tambahan acak, jumlah sama
    print(f"  Tambahan terarah: {n_extra} patch (hanya kebakaran dan gangguan alami)")
    print(f"  Kontrol         : {n_extra} patch (campuran kelas biasa), supaya adil: jumlah data\n"
          "                    bertambah sama banyak, hanya komposisinya yang berbeda.\n")

    def latih_dan_uji(nama, X, yy, seed=0):
        print(f"  Melatih model '{nama}' ({len(yy)} patch) ...")
        model, stat, _ = mc.latih(X, yy, va["X"], va["y"], epoch=18, seed=seed, verbose=False)
        p = mc.prediksi_proba(model, stat, te["X"]).argmax(1)
        return mc.metrik(y, p)

    dasar = mc.metrik(y, pred)
    m_terarah = latih_dan_uji("terarah", np.concatenate([tr["X"], extra_terarah["X"]]),
                              np.concatenate([tr["y"], extra_terarah["y"]]))
    m_kontrol = latih_dan_uji("kontrol", np.concatenate([tr["X"], extra_kontrol["X"]]),
                              np.concatenate([tr["y"], extra_kontrol["y"]]))

    print("  Mengukur variasi acak: model dasar dilatih ulang dengan dua seed lain ...")
    ulang = [dasar] + [latih_dan_uji(f"dasar seed {sd}", tr["X"], tr["y"], seed=sd) for sd in (1, 2)]
    rentang_mf1 = np.ptp([m["macro_f1"] for m in ulang])
    rentang_api = np.ptp([m["f1_per_kelas"][IDX_API] for m in ulang])
    rentang_alami = np.ptp([m["f1_per_kelas"][IDX_ALAMI] for m in ulang])

    skenario = {"Dasar": dasar, "+ Terarah": m_terarah, "+ Kontrol acak": m_kontrol}
    print(f"\n  {'model':16s}{'macro-F1':>10s}{'F1 kebakaran':>14s}{'F1 alami':>10s}")
    for nama, m in skenario.items():
        print(f"  {nama:16s}{m['macro_f1']:10.3f}{m['f1_per_kelas'][IDX_API]:14.3f}"
              f"{m['f1_per_kelas'][IDX_ALAMI]:10.3f}")
    print(f"\n  Variasi acak model dasar (3 seed): rentang macro-F1 {rentang_mf1:.3f} | "
          f"F1 kebakaran {rentang_api:.3f} | F1 alami {rentang_alami:.3f}")
    sel_api = m_terarah["f1_per_kelas"][IDX_API] - m_kontrol["f1_per_kelas"][IDX_API]
    sel_alami = m_terarah["f1_per_kelas"][IDX_ALAMI] - m_kontrol["f1_per_kelas"][IDX_ALAMI]
    print(f"  Selisih Terarah - Kontrol: F1 kebakaran {sel_api:+.3f} | F1 alami {sel_alami:+.3f}")
    if abs(sel_api) <= rentang_api and abs(sel_alami) <= rentang_alami:
        print("  Kesimpulan: selisih tidak lebih besar daripada variasi antar-seed, jadi pada percobaan\n"
              "  ini efek penambahan terarah TIDAK dapat dibedakan dari kebetulan. Pada data sintetis\n"
              "  ini model dasar sudah cukup baik pada kedua kelas; pada data nyata (kelas yang lebih\n"
              "  sulit dan label lebih berisik) hasilnya bisa berbeda, seperti yang dilaporkan pemateri.")
    else:
        print("  Kesimpulan: selisih lebih besar daripada variasi antar-seed pada setidaknya satu kelas;\n"
              "  periksa tandanya (+ berarti tambahan terarah lebih baik daripada kontrol).")
    hasil["retraining"] = {n: {k: v for k, v in m.items()} for n, m in skenario.items()}
    hasil["variasi_seed"] = {"macro_f1": float(rentang_mf1), "f1_kebakaran": float(rentang_api),
                             "f1_alami": float(rentang_alami)}

    nama_metrik = ["macro-F1", "F1 kebakaran", "F1 gangguan alami"]
    nilai = {
        n: [m["macro_f1"], m["f1_per_kelas"][IDX_API], m["f1_per_kelas"][IDX_ALAMI]]
        for n, m in skenario.items()
    }
    plt.figure(figsize=(7.5, 4))
    lebar = 0.26
    for i, (n, v) in enumerate(nilai.items()):
        plt.bar(np.arange(3) + (i - 1) * lebar, v, lebar, label=n)
    plt.xticks(range(3), nama_metrik)
    lo = min(min(v) for v in nilai.values())
    plt.ylim(max(0, lo - 0.1), 1.0)
    plt.ylabel("Skor (data uji)")
    plt.title("Pelatihan ulang terarah vs kontrol (sumbu y dipotong)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "retrain_terarah.png", dpi=130)
    plt.close()

    with open(OUT / "evaluasi_lanjutan.json", "w") as f:
        json.dump(hasil, f, indent=2, default=float)
    print("\nTersimpan: output/evaluasi_lanjutan.json, kompromi_keyakinan.png, retrain_terarah.png")


if __name__ == "__main__":
    main()

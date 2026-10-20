# Atribusi Pemicu Deforestasi dengan Citra Satelit dan CNN

Proyek praktik dari materi **"Attributing direct drivers to deforestation alerts with
satellite imagery and AI"** (Informatic Practitioner Lecture Series, pemateri peneliti dari
Universitas Wageningen; ejaan nama pemateri di transkrip tidak konsisten, jadi tidak saya tuliskan). Gagasan inti materi: peringatan deforestasi hanya menunjukkan *di mana*
tutupan pohon hilang, bukan *penyebab langsungnya*. Maka dibangun model deep learning (CNN) yang
membaca citra Sentinel-1 (radar) dan Sentinel-2 (optik) untuk mengklasifikasikan pemicunya
(pertanian, penebangan selektif, pertambangan, kebakaran, dan seterusnya).

## PENTING: data di proyek ini sintetis

Proyek ini **tidak memakai citra satelit asli**. Patch citra dibuat oleh kode (`data_sintetis.py`)
agar semua orang bisa menjalankannya tanpa akses ke Sentinel. Konsekuensinya:

- Angka akurasi **tidak mencerminkan** performa pada data nyata. Data sintetis ini jauh lebih mudah.
- Pola wilayah dan musiman sengaja ditanam di generator supaya analisisnya bisa dipraktikkan.
- Tujuh kelas pemicu disusun dari contoh yang disebut di materi (materi hanya menyebut ada tujuh
  kelas tanpa merinci semuanya). Pemilihan 4 band Sentinel-2 (biru, hijau, merah, NIR) adalah
  asumsi saya; materi hanya menyebut "empat band".

## Cara menjalankan

```bash
pip install -r requirements.txt
python main.py            # semua langkah, sekitar 6-8 menit di CPU
```

Atau satu langkah saja, berurutan (langkah selanjutnya butuh hasil langkah sebelumnya):

```bash
python 01_buat_dataset.py
python 02_baseline_random_forest.py
python 03_latih_cnn.py
python 04_evaluasi_lanjutan.py
python 05_analisis_musiman.py
python 06_prediksi_bulanan.py
```

Tidak butuh GPU. Folder `data/` dibuat otomatis oleh langkah 1; grafik dan hasil ada di `output/`.

## Langkah dan kaitannya dengan materi

| Langkah | File | Isi | Bagian materi |
|---|---|---|---|
| 1 | `01_buat_dataset.py` | Dataset sintetis 6 band (2 radar + 4 optik), split train/val/test, gambar contoh, distribusi kelas | Citra Sentinel-1 dan Sentinel-2 sebagai masukan; kelas tidak seimbang |
| 2 | `02_baseline_random_forest.py` | Baseline konvensional + ablasi sensor (hanya S1, hanya S2, gabungan) | Tanya jawab: coba metode konvensional dulu; kapan S1 vs S2 |
| 3 | `03_latih_cnn.py` | CNN PyTorch, accuracy dan macro-F1, F1 per kelas, confusion matrix, perbandingan dengan baseline | Klasifikasi pemicu dengan CNN; penilaian akurasi |
| 4 | `04_evaluasi_lanjutan.py` | Akurasi per wilayah, label "tidak yakin", pengaruh awan, kebakaran vs banjir, pelatihan ulang terarah dengan kontrol | Penilaian per wilayah; kelas tidak percaya diri; awan; pelatihan ulang kebakaran dan banjir |
| 5 | `05_analisis_musiman.py` | Indeks musiman per pemicu dan wilayah dari aliran peringatan | Analisis musiman pemicu |
| 6 | `06_prediksi_bulanan.py` | Klasifikasi batch bulanan, label "tidak yakin", keluaran CSV | Rilis bulanan, latensi, verifikasi lapangan |

Modul pendukung: `data_sintetis.py` (generator) dan `model_cnn.py` (arsitektur, pelatihan, prediksi).

## Hasil pada percobaan saya (data sintetis, satu kali jalan)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (S1 saja) | 0,730 | 0,674 |
| Random Forest (S2 saja) | 0,757 | 0,719 |
| Random Forest (S1+S2) | 0,822 | 0,786 |
| CNN (S1+S2) | 0,955 | 0,934 |

Temuan yang sejalan dengan materi:

- **Menggabungkan S1 dan S2 lebih baik** daripada salah satunya. Untuk kelas gangguan alami (banjir),
  S1 saja sedikit lebih baik daripada S2 saja (F1 0,674 vs 0,615), konsisten dengan penjelasan bahwa
  radar sensitif terhadap air.
- **Kelas kecil lebih sulit.** Pertanian skala besar F1-nya hampir sempurna, sedangkan jalan
  (0,813) dan penebangan selektif (0,883) paling lemah. Itu sebabnya macro-F1 dilaporkan di
  samping accuracy.
- **Kebakaran dan gangguan alami tertukar:** 11% kasus gangguan alami diprediksi kebakaran.
- **Ambang keyakinan 0,7** mengubah sekitar 6,5% prediksi data uji menjadi "tidak yakin",
  dan accuracy sisanya naik dari 0,955 menjadi 0,978. Kasus "tidak yakin" paling banyak berasal
  dari kelas jalan dan penebangan selektif.
- **Awan tebal** menurunkan accuracy (0,971 tanpa awan menjadi 0,921 pada awan tebal).

Temuan yang **tidak** terbukti:

- **Pelatihan ulang terarah** (menambah contoh kebakaran dan banjir) **tidak** mengungguli
  kontrol berupa tambahan data acak dengan jumlah sama. Selisihnya lebih kecil daripada variasi
  antar-seed, jadi tidak bisa dibedakan dari kebetulan. Pada data sintetis ini model dasar sudah
  cukup baik; pada data nyata hasilnya bisa berbeda, seperti yang dilaporkan pemateri.
- **Keunggulan CNN atas Random Forest** hanya satu percobaan, dan fitur manual Random Forest di sini
  sederhana. Fitur yang lebih baik dapat mempersempit selisih.

## Batasan yang perlu diingat (juga ditekankan di materi)

- Model hanya mengklasifikasikan **pemicu langsung** yang terlihat setelah gangguan. Penyebab
  sebenarnya bisa jauh lebih kompleks, dan memahaminya butuh data lapangan.
- Satu kelas bisa memuat banyak aktivitas yang tak terbedakan dari citra (contoh di materi:
  pertanian skala kecil, penebangan skala kecil, dan produksi arang di Lembah Kongo).
- Klasifikasi selalu bisa salah; untuk penegakan hukum atau intervensi perlu **verifikasi lapangan
  dan peninjauan manusia**. Rilis bulanan juga berarti ada latensi.
- Indeks musiman di langkah 5 adalah definisi buatan proyek ini (1 - entropi/ln 12, dengan koreksi
  sampel kecil), bukan definisi resmi dari pemateri.

## Ide pengembangan

1. **Pakai data nyata.** Materi menyebut citra Sentinel-1/2, peringatan deforestasi berbasis radar dan
   GLAD, platform Global Nature Watch, serta penggunaan Google Earth Engine oleh peserta. Mulai dari
   area kecil yang Anda kenal, dan siapkan label lapangan sendiri.
2. **Tangani ketidakseimbangan kelas** dengan bobot kelas pada loss, lalu bandingkan macro-F1.
3. **Ulangi percobaan dengan banyak seed** dan laporkan rata-rata serta sebarannya.
4. **Evaluasi per wilayah secara terpisah** (latih di satu wilayah, uji di wilayah lain) untuk menguji
   generalisasi.
5. **Tambahkan dimensi waktu**, yaitu tumpukan citra beberapa bulan, sesuai arah yang disebut
   di materi untuk melihat urutan kejadian antar-pemicu.
6. **Kalibrasi probabilitas** agar angka keyakinan lebih bermakna.

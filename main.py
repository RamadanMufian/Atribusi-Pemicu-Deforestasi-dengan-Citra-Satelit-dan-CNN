"""Menjalankan seluruh proyek berurutan: python main.py

Perkiraan waktu di CPU biasa: sekitar 6-8 menit (terutama langkah 3 dan 4 yang melatih CNN).
Untuk menjalankan satu langkah saja, jalankan skripnya langsung, misalnya:
    python 03_latih_cnn.py
"""
import subprocess
import sys
import time
from pathlib import Path

folder = Path(__file__).parent
langkah = [
    "01_buat_dataset.py",
    "02_baseline_random_forest.py",
    "03_latih_cnn.py",
    "04_evaluasi_lanjutan.py",
    "05_analisis_musiman.py",
    "06_prediksi_bulanan.py",
]
for skrip in langkah:
    t0 = time.time()
    subprocess.run([sys.executable, str(folder / skrip)], check=True, cwd=folder)
    print(f"\n[selesai {skrip} dalam {time.time() - t0:.0f} detik]\n\n")

"""
GENERATOR DATA SINTETIS - citra satelit multi-band untuk atribusi pemicu deforestasi.

Mengikuti gambaran di materi: setiap peringatan deforestasi diwakili potongan citra
(patch) yang menumpuk 6 band:
    2 band Sentinel-1 (radar)   : VV, VH
    4 band Sentinel-2 (optik)   : Biru, Hijau, Merah, NIR (inframerah dekat)
Model melihat tumpukan ini sebagai satu "citra multi-band".

PENTING: ini data SINTETIS buatan kode, bukan citra satelit asli. Nilai band, bentuk
gangguan, dan proporsi kelas hanyalah pendekatan sederhana agar proyek bisa dijalankan
tanpa akses ke data Sentinel. Hasil akurasinya TIDAK mencerminkan performa pada data nyata.

Tujuh kelas pemicu (daftar ini disusun dari contoh yang disebut di materi; materi hanya
menyebut ada tujuh kelas tanpa merinci seluruhnya):
    0 pertanian_skala_kecil   : beberapa bercak kecil
    1 pertanian_skala_besar   : lahan luas berbentuk bidang lurus
    2 jalan                   : garis tipis memanjang
    3 penebangan_selektif     : jalan tipis + celah-celah kecil di sekitarnya
    4 pertambangan            : bercak tak beraturan sangat terang, kadang ada kolam
    5 kebakaran               : area terbakar (NIR turun tajam, gelap)
    6 gangguan_alami          : banjir (pita air lebar di sungai) atau pohon tumbang
"""
import numpy as np
from scipy.ndimage import gaussian_filter

UKURAN = 32  # patch 32 x 32 piksel (di data asli satu piksel = 10 m)

KELAS = [
    "pertanian_skala_kecil",
    "pertanian_skala_besar",
    "jalan",
    "penebangan_selektif",
    "pertambangan",
    "kebakaran",
    "gangguan_alami",
]
BAND = ["S1_VV", "S1_VH", "S2_biru", "S2_hijau", "S2_merah", "S2_NIR"]
IDX_S1 = [0, 1]
IDX_S2 = [2, 3, 4, 5]

WILAYAH = ["Amerika Selatan", "Afrika Tengah", "Asia Tenggara"]
BOBOT_WILAYAH = [0.40, 0.35, 0.25]

# Komposisi kelas per wilayah (urutan sesuai KELAS). Terinspirasi cerita di materi:
# Amerika Selatan didominasi pertanian skala besar + kebakaran, Afrika Tengah oleh pertanian
# skala kecil + penebangan selektif, Asia Tenggara berupa campuran. Angkanya hanya ilustrasi.
CAMPURAN_KELAS = {
    "Amerika Selatan": [0.15, 0.35, 0.08, 0.07, 0.04, 0.26, 0.05],
    "Afrika Tengah":   [0.50, 0.05, 0.08, 0.20, 0.07, 0.06, 0.04],
    "Asia Tenggara":   [0.25, 0.25, 0.08, 0.17, 0.08, 0.10, 0.07],
}
# Puncak musim kering (bulan) dan kekuatan musiman per kelas (0 = merata, 1 = sangat terpusat).
PUNCAK_BULAN = {"Amerika Selatan": 8, "Afrika Tengah": 7, "Asia Tenggara": 9}
KEKUATAN_MUSIM = {
    "Amerika Selatan": [0.40, 0.50, 0.10, 0.15, 0.05, 0.75, 0.10],
    "Afrika Tengah":   [0.45, 0.30, 0.10, 0.15, 0.05, 0.65, 0.10],
    "Asia Tenggara":   [0.10, 0.10, 0.05, 0.05, 0.05, 0.25, 0.05],
}
# Peluang tutupan awan pada citra optik (lebih tinggi di daerah tropis basah).
PELUANG_AWAN = {"Amerika Selatan": 0.30, "Afrika Tengah": 0.45, "Asia Tenggara": 0.55}

# Nilai band (skala 0-1, sudah dinormalkan) untuk tiap tipe permukaan.
#                 VV    VH    biru  hijau merah NIR
HUTAN         = [0.55, 0.40, 0.10, 0.15, 0.10, 0.60]
TANAH_TERBUKA = [0.38, 0.20, 0.13, 0.22, 0.30, 0.33]
LAHAN_BESAR   = [0.35, 0.18, 0.13, 0.23, 0.32, 0.32]
TEBANG_CELAH  = [0.46, 0.30, 0.11, 0.18, 0.18, 0.46]
TAMBANG       = [0.72, 0.35, 0.22, 0.30, 0.38, 0.22]
KOLAM         = [0.06, 0.04, 0.09, 0.12, 0.08, 0.10]
TERBAKAR      = [0.46, 0.30, 0.07, 0.09, 0.13, 0.20]
AIR           = [0.12, 0.08, 0.10, 0.14, 0.08, 0.12]
POHON_TUMBANG = [0.46, 0.30, 0.10, 0.15, 0.17, 0.40]
AWAN          = [0.55, 0.40, 0.80, 0.82, 0.82, 0.78]  # hanya dipakai pada band optik

_YY, _XX = np.mgrid[0:UKURAN, 0:UKURAN].astype(np.float32)


# --------------------------------------------------------------------------- bentuk
def _medan(rng, sigma):
    f = gaussian_filter(rng.standard_normal((UKURAN, UKURAN)), sigma)
    return f / (f.std() + 1e-9)


def _blob(rng, cx, cy, r):
    """Bercak tak beraturan berpusat di (cx, cy) dengan jari-jari ~r."""
    d = np.hypot(_XX - cx, _YY - cy)
    return d < r * (1 + 0.30 * _medan(rng, 3.0))


def _garis(rng, lebar):
    """Garis lurus yang melintasi patch dengan sudut acak."""
    x0, y0 = rng.uniform(8, UKURAN - 8, 2)
    a = rng.uniform(0, np.pi)
    jarak = np.abs((_XX - x0) * np.sin(a) - (_YY - y0) * np.cos(a))
    return jarak < lebar / 2


def _persegi(rng, frac):
    """Bidang persegi panjang miring tipis, luas ~frac dari patch."""
    luas = frac * UKURAN * UKURAN
    rasio = rng.uniform(0.6, 1.6)
    w, h = np.sqrt(luas * rasio), np.sqrt(luas / rasio)
    cx, cy = rng.uniform(UKURAN * 0.3, UKURAN * 0.7, 2)
    a = rng.uniform(-0.35, 0.35)
    u = (_XX - cx) * np.cos(a) + (_YY - cy) * np.sin(a)
    v = -(_XX - cx) * np.sin(a) + (_YY - cy) * np.cos(a)
    return (np.abs(u) < w / 2) & (np.abs(v) < h / 2)


def _sungai(rng, lebar):
    """Pita berkelok yang melintasi patch (sungai)."""
    y0 = rng.uniform(8, UKURAN - 8)
    amp, lam = rng.uniform(2, 6), rng.uniform(16, 32)
    fase, miring = rng.uniform(0, 2 * np.pi), rng.uniform(-0.3, 0.3)
    kurva = y0 + amp * np.sin(2 * np.pi * _XX / lam + fase) + miring * (_XX - UKURAN / 2)
    return np.abs(_YY - kurva) < lebar / 2


def _terapkan(citra, mask, nilai, rng, jitter, tekstur=0.02, alpha=1.0):
    """Ganti nilai band di area mask dengan `nilai` (diberi jitter dan tekstur)."""
    m = gaussian_filter(mask.astype(np.float32), 0.7) * alpha
    nilai = np.asarray(nilai, np.float32)[:, None, None] * jitter[:, None, None]
    isi = nilai + tekstur * np.stack([_medan(rng, 1.5) for _ in range(6)])
    return citra * (1 - m) + isi * m


# --------------------------------------------------------------------------- satu patch
def buat_patch(kelas, wilayah, rng):
    """Membuat satu patch (6, 32, 32) untuk kelas pemicu tertentu. Mengembalikan (citra, frac_awan)."""
    jitter_hutan = rng.normal(1.0, 0.07, 6)
    dasar = np.asarray(HUTAN, np.float32)[:, None, None] * jitter_hutan[:, None, None]
    citra = dasar + 0.03 * np.stack([_medan(rng, 2.0) for _ in range(6)])

    # Konteks: sungai sempit hadir di sebagian patch apa pun (supaya banjir tidak "terlalu mudah")
    if rng.random() < 0.25:
        citra = _terapkan(citra, _sungai(rng, rng.uniform(2, 3)), AIR, rng, np.ones(6))

    jit = rng.normal(1.0, 0.13, 6)  # variasi nilai efek antar-kejadian
    cx, cy = rng.uniform(10, UKURAN - 10, 2)

    if kelas == 0:      # pertanian skala kecil: beberapa bercak kecil berdekatan
        for _ in range(rng.integers(2, 5)):
            bx, by = cx + rng.normal(0, 4), cy + rng.normal(0, 4)
            citra = _terapkan(citra, _blob(rng, bx, by, rng.uniform(2.0, 4.0)),
                              TANAH_TERBUKA, rng, jit, alpha=rng.uniform(0.7, 1.0))
    elif kelas == 1:    # pertanian skala besar: bidang luas
        citra = _terapkan(citra, _persegi(rng, rng.uniform(0.25, 0.50)),
                          LAHAN_BESAR, rng, jit, tekstur=0.01)
    elif kelas == 2:    # jalan: garis tipis
        citra = _terapkan(citra, _garis(rng, rng.uniform(1.5, 2.5)), TANAH_TERBUKA, rng, jit)
    elif kelas == 3:    # penebangan selektif: jalan tipis + celah-celah kecil
        garis = _garis(rng, rng.uniform(1.2, 2.0))
        citra = _terapkan(citra, garis, TANAH_TERBUKA, rng, jit, alpha=0.8)
        for _ in range(rng.integers(3, 7)):
            bx, by = rng.uniform(4, UKURAN - 4, 2)
            citra = _terapkan(citra, _blob(rng, bx, by, rng.uniform(1.2, 2.5)),
                              TEBANG_CELAH, rng, jit)
    elif kelas == 4:    # pertambangan: bercak terang besar, kadang ada kolam
        citra = _terapkan(citra, _blob(rng, cx, cy, rng.uniform(6, 9)),
                          TAMBANG, rng, jit, tekstur=0.04)
        if rng.random() < 0.5:
            citra = _terapkan(citra, _blob(rng, cx + rng.normal(0, 3), cy + rng.normal(0, 3),
                                           rng.uniform(2, 3.5)), KOLAM, rng, jit)
    elif kelas == 5:    # kebakaran: area terbakar, sering dekat sungai
        if rng.random() < 0.35:
            citra = _terapkan(citra, _sungai(rng, rng.uniform(2, 3)), AIR, rng, np.ones(6))
        citra = _terapkan(citra, _blob(rng, cx, cy, rng.uniform(5, 9)), TERBAKAR, rng, jit)
    else:               # gangguan alami: banjir (pita air lebar) atau pohon tumbang
        if rng.random() < 0.6:
            citra = _terapkan(citra, _sungai(rng, rng.uniform(4, 9)), AIR, rng, jit)
        else:
            citra = _terapkan(citra, _blob(rng, cx, cy, rng.uniform(4, 8)),
                              POHON_TUMBANG, rng, jit)

    # Speckle pada radar (multiplikatif), derau aditif pada optik
    citra[IDX_S1] *= 1 + 0.10 * rng.standard_normal((2, UKURAN, UKURAN))
    citra[IDX_S2] += 0.015 * rng.standard_normal((4, UKURAN, UKURAN))

    # Tutupan awan hanya memengaruhi band optik
    frac_awan = 0.0
    if rng.random() < PELUANG_AWAN[wilayah]:
        target = rng.uniform(0.3, 1.0)
        f = _medan(rng, 5.0)
        awan = f > np.quantile(f, 1 - target)
        m = gaussian_filter(awan.astype(np.float32), 1.0)
        for b in IDX_S2:
            citra[b] = citra[b] * (1 - m) + AWAN[b] * m
        frac_awan = float(awan.mean())

    return np.clip(citra, 0, 1).astype(np.float32), frac_awan


# --------------------------------------------------------------------------- dataset
def _prob_bulan(wilayah, kelas):
    m = np.arange(1, 13)
    puncak = np.exp(2.0 * np.cos(2 * np.pi * (m - PUNCAK_BULAN[wilayah]) / 12))
    puncak /= puncak.sum()
    s = KEKUATAN_MUSIM[wilayah][kelas]
    return (1 - s) / 12 + s * puncak


def buat_dataset(n, seed, hanya_kelas=None):
    """Membuat dataset berisi n patch.

    hanya_kelas: daftar indeks kelas, jika diisi hanya kelas itu yang dibuat. Dipakai untuk
    menambah contoh pelatihan kelas yang sulit (kebakaran dan banjir) seperti di materi.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 6, UKURAN, UKURAN), np.float32)
    y = np.zeros(n, np.int64)
    wil = np.zeros(n, np.int64)
    bulan = np.zeros(n, np.int64)
    awan = np.zeros(n, np.float32)
    for i in range(n):
        w = rng.choice(len(WILAYAH), p=BOBOT_WILAYAH)
        p = np.array(CAMPURAN_KELAS[WILAYAH[w]], dtype=float)
        if hanya_kelas is not None:
            mask = np.zeros(len(KELAS), bool)
            mask[list(hanya_kelas)] = True
            p = p * mask
        k = rng.choice(len(KELAS), p=p / p.sum())
        X[i], awan[i] = buat_patch(k, WILAYAH[w], rng)
        y[i], wil[i] = k, w
        bulan[i] = rng.choice(np.arange(1, 13), p=_prob_bulan(WILAYAH[w], k))
    return {"X": X, "y": y, "wilayah": wil, "bulan": bulan, "awan": awan}


def simpan(path, data):
    np.savez_compressed(path, **data)


def muat(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}

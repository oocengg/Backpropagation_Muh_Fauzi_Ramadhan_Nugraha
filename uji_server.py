"""Server uji lokal: halaman web untuk menguji jaringan back propagation dengan gambar sendiri.

Jalankan dari folder ini:
    .venv/bin/python uji_server.py            (atau: python uji_server.py)
lalu buka alamat yang tercetak (biasanya http://127.0.0.1:8765). Server hanya dapat diakses dari
komputer ini (127.0.0.1), dan gambar yang diunggah hanya diproses di memori, tidak disimpan.

Dua model tersedia di halaman:
  - siluet   : with_silhouette/bobot.npz, preprocessing GrabCut -> siluet objek 100 x 100 (model utama)
  - baseline : with_grayscale/bobot.npz, preprocessing grayscale -> resize 100 x 100
Preprocessing dan propagasi maju identik dengan training.ipynb / pengujian.ipynb (notasi buku Fausett).
"""
import argparse
import base64
import io
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
UKURAN = (100, 100)           # tinggi x lebar
AMBANG = 0.5                  # output terbesar di bawah ambang -> tidak dikenali
MAKS_BYTE = 25 * 1024 * 1024


def f(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- preprocessing
def praproses_siluet(gambar):
    # Persis sama dengan notebook: GrabCut dengan bingkai awal 92% bagian tengah, siluet 0/255, resize 100 x 100.
    rgb = np.asarray(gambar.convert("RGB"))
    tinggi, lebar = rgb.shape[:2]
    skala = 200 / max(tinggi, lebar)
    kecil = cv2.resize(rgb, (int(lebar * skala), int(tinggi * skala)), interpolation=cv2.INTER_AREA)
    bgr = kecil[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    bingkai = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
    topeng = np.zeros((h, w), np.uint8)
    model_latar, model_objek = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.setRNGSeed(0)
    cv2.grabCut(bgr, topeng, bingkai, model_latar, model_objek, 5, cv2.GC_INIT_WITH_RECT)
    objek = (topeng == cv2.GC_FGD) | (topeng == cv2.GC_PR_FGD)
    citra = Image.fromarray((objek * 255).astype(np.uint8)).resize((UKURAN[1], UKURAN[0]), Image.LANCZOS)
    peringatan = None
    if objek.mean() < 0.02:
        peringatan = "Objek hampir tidak terdeteksi. Gunakan foto dengan satu objek utama di dalam bingkai."
    return citra, peringatan


def praproses_baseline(gambar):
    citra = gambar.convert("L").resize((UKURAN[1], UKURAN[0]), Image.LANCZOS)
    return citra, None


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, kunci, judul, berkas, praproses):
        self.kunci, self.judul, self.praproses = kunci, judul, praproses
        b = np.load(berkas)
        self.v, self.v_0, self.w, self.w_0 = b["v"], b["v_0"], b["w"], b["w_0"]
        self.nama_kelas = [str(k) for k in b["nama_kelas"]]
        self.n = self.v.shape[0]

    def prediksi(self, gambar):
        citra, peringatan = self.praproses(gambar)
        x = np.asarray(citra, dtype=np.float64).flatten() / 255.0
        assert x.shape == (self.n,)
        z = f(self.v_0 + x @ self.v)                  # z_j = f(v_0j + sum_i x_i v_ij)
        y = f(self.w_0 + z @ self.w)                  # y_k = f(w_0k + sum_j z_j w_jk)
        k = int(np.argmax(y))
        buf = io.BytesIO()
        citra.save(buf, format="PNG")
        return {
            "model": self.kunci, "judul": self.judul, "y": [float(v) for v in y], "kelas": k,
            "nama_kelas": self.nama_kelas, "dikenali": bool(y[k] >= AMBANG), "ambang": AMBANG,
            "gambar": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "peringatan": peringatan,
        }


def muat_model():
    daftar = {}
    siluet = BASE / "with_silhouette" / "bobot.npz"
    grayscale = BASE / "with_grayscale" / "bobot.npz"
    if siluet.exists():
        daftar["siluet"] = Model("siluet", "Siluet objek (GrabCut)", siluet, praproses_siluet)
    if grayscale.exists():
        daftar["baseline"] = Model("baseline", "Grayscale (baseline)", grayscale, praproses_baseline)
    if not daftar:
        raise SystemExit("Tidak ada file bobot.npz di with_silhouette/ maupun with_grayscale/. Jalankan training dulu.")
    return daftar


MODEL = {}


# ---------------------------------------------------------------- HTTP
class Penangan(BaseHTTPRequestHandler):
    def _kirim(self, kode, isi, jenis):
        self.send_response(kode)
        self.send_header("Content-Type", jenis)
        self.send_header("Content-Length", str(len(isi)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(isi)

    def _json(self, kode, obj):
        self._kirim(kode, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        jalur = urlparse(self.path).path
        if jalur in ("/", "/uji.html"):
            self._kirim(200, (BASE / "uji.html").read_bytes(), "text/html; charset=utf-8")
        elif jalur == "/model":
            self._json(200, {k: m.judul for k, m in MODEL.items()})
        else:
            self._kirim(404, b"tidak ditemukan", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/prediksi":
            return self._kirim(404, b"tidak ditemukan", "text/plain; charset=utf-8")
        pilihan = parse_qs(parsed.query).get("model", ["siluet"])[0]
        kunci = list(MODEL) if pilihan == "keduanya" else [pilihan]
        if any(k not in MODEL for k in kunci):
            return self._json(400, {"galat": f"Model tidak dikenal: {pilihan}"})
        panjang = int(self.headers.get("Content-Length", 0))
        if panjang <= 0 or panjang > MAKS_BYTE:
            return self._json(400, {"galat": "Ukuran gambar tidak valid (maksimum 25 MB)."})
        data = self.rfile.read(panjang)
        try:
            gambar = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))     # foto ponsel: putar sesuai kamera
            gambar.load()
        except Exception as e:
            return self._json(400, {"galat": f"Berkas bukan gambar yang dapat dibaca ({e.__class__.__name__})."})
        try:
            self._json(200, {"hasil": [MODEL[k].prediksi(gambar) for k in kunci]})
        except Exception as e:
            self._json(500, {"galat": f"Gagal memproses gambar: {e}"})

    def log_message(self, fmt, *args):         # log singkat tanpa menulis isi permintaan
        print("  ", self.address_string(), fmt % args)


def port_bebas(awal):
    for p in range(awal, awal + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("Tidak ada port bebas.")


def main():
    ap = argparse.ArgumentParser(description="Server uji back propagation (lokal)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tanpa-browser", action="store_true", help="jangan buka browser otomatis")
    a = ap.parse_args()
    MODEL.update(muat_model())
    port = port_bebas(a.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), Penangan)
    url = f"http://127.0.0.1:{port}"
    print(f"Model dimuat: {', '.join(m.judul for m in MODEL.values())}")
    print(f"Buka {url}   (tekan Ctrl+C untuk berhenti)")
    if not a.tanpa_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")


if __name__ == "__main__":
    main()

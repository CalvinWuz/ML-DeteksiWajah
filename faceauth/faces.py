"""Deteksi dan potong wajah (OpenCV YuNet).

Semua sumber data (FFHQ/StyleGAN, DeepFakeFace, OpenFake, gambar buatan pengguna) dan demo
memakai fungsi yang sama, supaya framing wajah seragam dan model tidak belajar dari perbedaan
cara memotong gambar.
"""
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .utils import PROJECT_ROOT

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = PROJECT_ROOT / "models_ext/face_detection_yunet_2023mar.onnx"
CROP_SIZE = 256
# FFHQ: tinggi kotak wajah YuNet ~0,8 x sisi gambar -> sisi potongan = 1,25 x sisi terpanjang kotak wajah
MARGIN = 1.25
MIN_FACE = 64  # piksel; wajah lebih kecil dari ini terlalu buram untuk dinilai

_local = threading.local()


def _detector():
    if not YUNET_PATH.exists():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    if not hasattr(_local, "det"):  # satu detektor per thread (YuNet tidak thread-safe)
        _local.det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.8, 0.3, 5000)
    return _local.det


def detect_faces(bgr: np.ndarray) -> list:
    """Kotak wajah [x, y, w, h, skor], urut dari yang terbesar."""
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > 1600:  # percepat deteksi pada gambar besar
        scale = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    det = _detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    boxes = [[f[0] / scale, f[1] / scale, f[2] / scale, f[3] / scale, float(f[14])] for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_box(box, img_w, img_h, margin=MARGIN):
    """Kotak persegi berpusat di wajah, digeser agar tetap di dalam gambar. None jika tidak muat."""
    x, y, w, h = box[:4]
    side = int(round(max(w, h) * margin))
    if side > min(img_w, img_h):
        side = min(img_w, img_h)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(min(max(cx - side / 2, 0), img_w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), img_h - side)))
    return x0, y0, side


def crop_largest_face(bgr: np.ndarray, size=CROP_SIZE, min_face=MIN_FACE):
    """Potong wajah terbesar -> (crop size x size BGR, info) atau (None, alasan)."""
    faces = detect_faces(bgr)
    if not faces:
        return None, "tidak ada wajah"
    box = faces[0]
    if max(box[2], box[3]) < min_face:
        return None, f"wajah terlalu kecil ({int(max(box[2], box[3]))} px)"
    h, w = bgr.shape[:2]
    x0, y0, side = crop_box(box, w, h)
    crop = bgr[y0:y0 + side, x0:x0 + side]
    interp = cv2.INTER_AREA if side >= size else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (size, size), interpolation=interp)
    info = {"n_faces": len(faces), "face_px": int(max(box[2], box[3])), "face_score": round(box[4], 3),
            "src_w": w, "src_h": h}
    return crop, info


def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def save_png(bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)  # PNG lossless: tidak menambah jejak kompresi

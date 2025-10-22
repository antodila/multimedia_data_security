# attacks.py
# Permitted attacks for the challenge.

import numpy as np
import cv2

def attack_jpeg(img, qf=90):
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)]
    ok, buf = cv2.imencode(".jpg", img, enc)
    if not ok: return img
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def attack_awgn(img, sigma=5):
    g = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + g
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_blur(img, ksize=3):
    k = ksize if ksize % 2 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def attack_median(img, ksize=3):
    k = ksize if ksize % 2 else ksize + 1
    return cv2.medianBlur(img, k)

def attack_resize(img, scale=0.6):
    h, w = img.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def attack_sharp(img, amount=1.0):
    blur = cv2.GaussianBlur(img, (0,0), 3)
    out = cv2.addWeighted(img.astype(np.float32), 1 + amount, blur.astype(np.float32), -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

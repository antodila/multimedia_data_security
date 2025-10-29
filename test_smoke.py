# test_smoke.py
import cv2
import numpy as np

from embedding import embedding
from detection_shadowmark import (
    _extract_ratio_vector, _fixed_coords, _winsorize,
    _band_masks, _calibrate_per_band, wpsnr,
    SEED, MARK_SIZE, MID_LO, MID_HI,
)
import attacks as A

IMG  = "lena_grey.bmp"
MARK = "shadowmark.npy"

def _weights_for_locs(locs):
    w = []
    for (u, v) in locs:
        s = u + v
        if s <= 15:   w.append(1.50)  # low
        elif s <= 50: w.append(1.10)  # mid
        else:         w.append(0.40)  # high
    return np.asarray(w, dtype=np.float32)

def similarity_weighted(a, b, w):
    n = min(a.size, b.size, w.size)
    a = a[:n].astype(np.float32)
    b = b[:n].astype(np.float32)
    w = w[:n]
    # z-score
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    num = float(np.sum(w * a * b))
    den = (np.sqrt(float(np.sum(w * a * a))) *
           np.sqrt(float(np.sum(w * b * b))) + 1e-12)
    return num / den

# --- load images and prepare ---
orig = cv2.imread(IMG, 0)
if orig is None:
    raise FileNotFoundError(f"Missing {IMG}")

wm_img = embedding(IMG, MARK)

H, W = orig.shape
locs  = _fixed_coords(H, W, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
wfreq = _weights_for_locs(locs)
low_idx, mid_idx, high_idx = _band_masks(locs)

# --- feature vectors (prefilter inside _extract_ratio_vector) ---
v_wm    = _extract_ratio_vector(wm_img, orig, locs)
v_wm    = _winsorize(v_wm, p=2.0)

v_clean = _extract_ratio_vector(orig, orig, locs)
v_clean = _winsorize(v_clean, p=2.0)

print("TP self-sim (≈1):", similarity_weighted(v_wm, v_wm, wfreq))
print("TN sim (wm vs clean):", similarity_weighted(v_wm, v_clean, wfreq))

# --- attacks battery ---
candidates = [
    ("jpeg",   {"qf": 70}),
    ("awgn",   {"sigma": 6}),
    ("blur",   {"ksize": 5}),
    ("median", {"ksize": 3}),
    ("resize", {"scale": 0.6}),
]

for name, params in candidates:
    if name == "jpeg":   att = A.attack_jpeg(wm_img, **params)
    if name == "awgn":   att = A.attack_awgn(wm_img, **params)
    if name == "blur":   att = A.attack_blur(wm_img, **params)
    if name == "median": att = A.attack_median(wm_img, **params)
    if name == "resize": att = A.attack_resize(wm_img, **params)

    v_att = _extract_ratio_vector(att, orig, locs)
    v_att = _winsorize(v_att, p=2.0)
    # per-band gain calibration
    v_att_cal = _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))

    sim_w = similarity_weighted(v_wm, v_att_cal, wfreq)

    w_wm = wpsnr(wm_img, att)   # WPSNR vs watermarked
    w_or = wpsnr(orig,   att)   # WPSNR vs original

    print(f"{name} {params} -> sim={sim_w:.3f}, WPSNR(wm,att)={w_wm:.2f}, WPSNR(orig,att)={w_or:.2f}")

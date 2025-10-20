# test_smoke.py
import cv2
from embed_spread_spectrum import embedding as embed_core
from detection_group import _extract_ratio_vector, _similarity, wpsnr
import attacks as A

IMG  = "lena_grey.bmp"
MARK = "mark.npy"

orig = cv2.imread(IMG, 0)
wm   = embed_core(IMG, MARK)

v_wm    = _extract_ratio_vector(wm, orig)
v_clean = _extract_ratio_vector(orig, orig)  # ~0 vector

print("TP self-sim (≈1):", _similarity(v_wm, v_wm))
print("TN sim (wm vs clean) (low):", _similarity(v_wm, v_clean))

for name, params in [
    ("jpeg",   {"qf": 60}),
    ("awgn",   {"sigma": 6}),
    ("blur",   {"ksize": 5}),
    ("median", {"ksize": 3}),
    ("resize", {"scale": 0.6}),
]:
    if name == "jpeg":   att = A.attack_jpeg(wm, **params)
    if name == "awgn":   att = A.attack_awgn(wm, **params)
    if name == "blur":   att = A.attack_blur(wm, **params)
    if name == "median": att = A.attack_median(wm, **params)
    if name == "resize": att = A.attack_resize(wm, **params)

    sim = _similarity(v_wm, _extract_ratio_vector(att, orig))
    wps = wpsnr(wm, att)
    print(f"{name} {params} -> sim={sim:.3f}, WPSNR≈{wps:.2f}")

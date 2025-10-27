import cv2
from embedding import embedding
from detection_shadowmark import _extract_ratio_vector, _similarity, wpsnr
import attacks as A

IMG  = "lena_grey.bmp"
MARK = "shadowmark.npy"   

orig = cv2.imread(IMG, 0)
if orig is None:
    raise FileNotFoundError(f"Missing {IMG}")
wm = embedding(IMG, MARK)

v_wm = _extract_ratio_vector(wm, orig)
print("TP self-sim (≈1):", _similarity(v_wm, v_wm))

v_clean = _extract_ratio_vector(orig, orig)  # ~0 vector
print("TN sim (wm vs clean):", _similarity(v_wm, v_clean))

candidates = [
    ("jpeg",   {"qf": 70}),
    ("awgn",   {"sigma": 6}),
    ("blur",   {"ksize": 5}),
    ("median", {"ksize": 3}),
    ("resize", {"scale": 0.6}),
]

for name, params in candidates:
    if name == "jpeg":   att = A.attack_jpeg(wm, **params)
    if name == "awgn":   att = A.attack_awgn(wm, **params)
    if name == "blur":   att = A.attack_blur(wm, **params)
    if name == "median": att = A.attack_median(wm, **params)
    if name == "resize": att = A.attack_resize(wm, **params)

    sim = _similarity(v_wm, _extract_ratio_vector(att, orig))
    w_wm  = wpsnr(wm, att)    # WPSNR vs watermarked 
    w_org = wpsnr(orig, att)  # WPSNR vs original 

    print(f"{name} {params} -> sim={sim:.3f}, WPSNR(wm,att)={w_wm:.2f}, WPSNR(orig,att)={w_org:.2f}")

# roc_threshold.py
import json, numpy as np, cv2
from sklearn.metrics import roc_curve, auc

# use your existing functions
from embed_spread_spectrum import embedding as embed_core  # 2-arg version: (input1, input2)
from detection_group import _extract_ratio_vector, _similarity

# === CONFIG ===
IMAGES = ["lena_grey.bmp"]   # add more images if you have them
MARK   = "mark.npy"          # competition watermark or dummy
# Light attacks intended to keep the mark (positives)
import attacks as A
ATT_LIGHT = [
    ("jpeg",   {"qf": 95}),
    ("jpeg",   {"qf": 90}),
    ("jpeg",   {"qf": 85}),
    ("blur",   {"ksize": 3}),
    ("median", {"ksize": 3}),
]
# Negatives: clean image and random vector

def do_attack(img, name, params):
    if name == "jpeg":   return A.attack_jpeg(img, **params)
    if name == "awgn":   return A.attack_awgn(img, **params)
    if name == "blur":   return A.attack_blur(img, **params)
    if name == "median": return A.attack_median(img, **params)
    if name == "resize": return A.attack_resize(img, **params)
    if name == "sharp":  return A.attack_sharp(img, **params)
    return img

scores, labels = [], []

for path in IMAGES:
    # Original, watermarked and its residual feature
    orig = cv2.imread(path, 0)
    wm   = embed_core(path, MARK)
    v_wm = _extract_ratio_vector(wm, orig)

    # --- True Positives: WM vs lightly-attacked WM (watermark should remain) ---
    for name, params in ATT_LIGHT:
        att = do_attack(wm, name, params)
        v_att = _extract_ratio_vector(att, orig)
        scores.append(_similarity(v_wm, v_att)); labels.append(1)

    # --- True Negatives: WM vs clean, and WM vs random ---
    v_clean = _extract_ratio_vector(orig, orig)        # ≈ 0 vector
    scores.append(_similarity(v_wm, v_clean)); labels.append(0)

    rand = np.random.randn(v_wm.size).astype(np.float32)
    # normalize rand like a feature vector (zero-mean)
    scores.append(_similarity(v_wm, rand)); labels.append(0)

scores = np.array(scores, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# ROC + AUC
fpr, tpr, thr = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Choose threshold: FPR ≤ 0.1 and maximum TPR
valid = np.where(fpr <= 0.1)[0]
idx = valid[np.argmax(tpr[valid])] if valid.size else np.argmin(fpr)
tau = float(thr[idx])

with open("tau.json", "w") as f:
    json.dump({"tau": tau, "FPR": float(fpr[idx]), "TPR": float(tpr[idx]), "AUC": float(roc_auc)}, f)

print(f"AUC={roc_auc:.3f}  tau={tau:.6f}  @ FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}")

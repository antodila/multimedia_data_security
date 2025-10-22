# roc_threshold.py
# Builds ROC using residual-ratio similarity and chooses tau (FPR<=0.1).
# Saves roc.png, roc.pdf, and tau.json.

import json, numpy as np, cv2
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from embedding import embedding              # our 2-arg embedding
# We re-implement ratio feature here to keep ROC independent of detection file.
from scipy.fft import dct
import attacks as A

# --- Config ---
IMAGES = ["lena_grey.bmp"]      # add your training images here
MARK   = "mark.npy"             # official or dummy
SEED = 123
MARK_SIZE = 1024
MID_LO, MID_HI = 4, 60

# --- Helpers (must mirror detection/embedding) ---
def _2d_dct(x):  return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    rng_local = np.random.default_rng(seed)
    coords = _midband_coords(h, w, lo, hi)
    while len(coords) < k and hi < (h + w - 2):
        hi += 2
        coords = _midband_coords(h, w, lo, hi)
    if len(coords) < k:
        k = len(coords)
    idx = rng_local.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

def extract_ratio_vector(img, orig):
    C_img  = _2d_dct(img.astype(np.float32))
    C_orig = _2d_dct(orig.astype(np.float32))
    eps = 1e-6
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig)*eps, C_orig)
    locs = _fixed_coords(*C_img.shape, k=MARK_SIZE, seed=SEED)
    return np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs],
                    dtype=np.float32)

def similarity(a, b):
    n = min(a.size, b.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32)
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    return float(np.sum(a*b)/n)

# Light attacks intended to preserve watermark (positives)
ATT_LIGHT = [
    ("jpeg",   {"qf": 95}),
    ("jpeg",   {"qf": 90}),
    ("jpeg",   {"qf": 85}),
    ("blur",   {"ksize": 3}),
    ("median", {"ksize": 3}),
]

def do_attack(img, name, params):
    if name == "jpeg":   return A.attack_jpeg(img, **params)
    if name == "awgn":   return A.attack_awgn(img, **params)
    if name == "blur":   return A.attack_blur(img, **params)
    if name == "median": return A.attack_median(img, **params)
    if name == "resize": return A.attack_resize(img, **params)
    if name == "sharp":  return A.attack_sharp(img, **params)
    return img

# --- Collect scores/labels ---
scores, labels = [], []
for path in IMAGES:
    orig = cv2.imread(path, 0)
    wm   = embedding(path, MARK)
    v_wm = extract_ratio_vector(wm, orig)

    # Positives: watermark preserved (light attacks)
    for name, params in ATT_LIGHT:
        att = do_attack(wm, name, params)
        v_att = extract_ratio_vector(att, orig)
        scores.append(similarity(v_wm, v_att)); labels.append(1)

    # Negatives: clean and random
    v_clean = extract_ratio_vector(orig, orig)  # ~0 vector
    scores.append(similarity(v_wm, v_clean)); labels.append(0)
    rand = np.random.randn(v_wm.size).astype(np.float32)
    scores.append(similarity(v_wm, rand)); labels.append(0)

scores = np.array(scores, np.float32)
labels = np.array(labels, np.int32)

# --- ROC & tau selection ---
fpr, tpr, thr = roc_curve(labels, scores)
the_auc = auc(fpr, tpr)
valid = np.where(fpr <= 0.1)[0]
idx = valid[np.argmax(tpr[valid])] if valid.size else np.argmin(fpr)
tau = float(thr[idx])

with open("tau.json", "w") as f:
    json.dump({"tau": tau, "FPR": float(fpr[idx]), "TPR": float(tpr[idx]), "AUC": float(the_auc)}, f)
print(f"AUC={the_auc:.3f}  tau={tau:.6f}  @ FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}")

# --- Save ROC plot ---
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC={the_auc:.3f}")
plt.plot([0,1],[0,1],"--", color="gray", lw=1)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc.png", dpi=300)
plt.savefig("roc.pdf")
plt.close()

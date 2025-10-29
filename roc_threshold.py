import json, glob, os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")          # headless backend (no GUI needed to save PNG/PDF)
import matplotlib.pyplot as plt
from embedding import embedding
from scipy.fft import dct
import attacks as A

# ----------------------------
# Dataset / embedding settings
# ----------------------------
IMG_FOLDER = "images"          # folder with 0000.bmp … 0100.bmp
MARK = "shadowmark.npy"        # official watermark
SEED = 123                     # fixed seed for reproducibility
MARK_SIZE = 1024               # number of DCT coords sampled
MID_LO, MID_HI = 1, 100        # (u+v) band used to choose coords

IMAGES = sorted(glob.glob(os.path.join(IMG_FOLDER, "0*.bmp")))
print(f"Found {len(IMAGES)} images for ROC computation.")

# ----------------------------
# Robust feature pre-processing
# ----------------------------
def _prefilter(img):
    """Light denoising before DCT to reduce small JPEG/resize artifacts."""
    return cv2.GaussianBlur(img, (3,3), 0.5)

def _winsorize(x, p=2.0):
    """
    Clamp extreme tails of x to the [p, 100-p] percentiles.
    This limits the influence of outliers in the similarity.
    """
    lo, hi = np.percentile(x, p), np.percentile(x, 100.0 - p)
    return np.clip(x, lo, hi)

def _band_masks(locs):
    """
    Build index masks (low/mid/high) based on (u+v).
    Used by the per-band gain calibration step.
    """
    low = []; mid = []; high = []
    for idx, (u,v) in enumerate(locs):
        s = u+v
        if s <= 15:      low.append(idx)
        elif s <= 50:    mid.append(idx)
        else:            high.append(idx)
    return np.array(low), np.array(mid), np.array(high)

def _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0)):
    """
    Per-band linear gain calibration:
    For each band B, estimate g_B that best aligns v_att[B] to v_wm[B],
    then clamp g_B to [clip] to avoid overfitting.
    This reduces benign frequency-dependent drifts from JPEG/resize.
    """
    eps = 1e-9
    def gain(a, b):
        num = float(np.dot(a, b))
        den = float(np.dot(b, b)) + eps
        g = num / den if den > 0 else 1.0
        return float(np.clip(g, clip[0], clip[1]))
    v_att_cal = v_att.copy()
    for idxs in (low_idx, mid_idx, high_idx):
        if idxs.size == 0: 
            continue
        g = gain(v_wm[idxs], v_att[idxs])
        v_att_cal[idxs] = g * v_att[idxs]
    return v_att_cal

def _weights_for_locs(locs):
    """
    Frequency weights for the weighted cosine similarity:
    up-weight low/mid (more stable watermark energy), down-weight high.
    """
    w = []
    for (u,v) in locs:
        s = u+v
        if s <= 15:    w.append(1.50)  # previously 1.30
        elif s <= 50:  w.append(1.10)  # previously 1.00
        else:          w.append(0.40)  # previously 0.60
    return np.asarray(w, dtype=np.float32)

def similarity_weighted(a, b, w):
    """
    Weighted cosine similarity with z-score standardization.
    Higher is better. Returns in [-1, 1].
    """
    n = min(a.size, b.size, w.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32); w = w[:n]
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    num = float(np.sum(w * a * b))
    den = np.sqrt(float(np.sum(w * a*a))) * np.sqrt(float(np.sum(w * b*b))) + 1e-12
    return num / den

# ----------------------------
# DCT coordinate helpers
# ----------------------------
def _2d_dct(x): 
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    """All (u,v) with lo <= (u+v) <= hi."""
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Deterministic random subset of size k from the mid-band region.
    Expands hi if needed to gather enough coordinates.
    """
    rng_local = np.random.default_rng(seed)
    coords = _midband_coords(h, w, lo, hi)
    while len(coords) < k and hi < (h + w - 2):
        hi += 2
        coords = _midband_coords(h, w, lo, hi)
    if len(coords) < k:
        k = len(coords)
    idx = rng_local.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

def extract_ratio_vector(img, orig, locs):
    """
    Feature: R(img) = C(img)/C(orig) - 1, evaluated at 'locs'.
    - Apply light prefiltering to 'img' to stabilize the ratio.
    - Keep 'orig' unfiltered as the clean reference spectrum.
    """
    C_img  = _2d_dct(_prefilter(img).astype(np.float32))
    C_orig = _2d_dct(orig.astype(np.float32))
    eps = 1e-6
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig)*eps, C_orig)
    return np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs], dtype=np.float32)

def similarity(a, b):
    """Unweighted cosine similarity with z-score (kept for diagnostics if needed)."""
    n = min(a.size, b.size)
    a = a[:n].astype(np.float32)
    b = b[:n].astype(np.float32)
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    return float(np.sum(a*b)/n)

# ----------------------------
# Attack configurations
# ----------------------------
# Positives: lightly attacked watermarked images should be detected
ATT_LIGHT = [
    ("jpeg",   {"qf": 90}),
    ("blur",   {"ksize": 3}),
    ("resize", {"scale": 0.9}),
]

# Negatives: harder attacks (on the watermarked image) that should be rejected
NEG_HARD = [
    ("awgn",   {"sigma": 10}),
    ("awgn",   {"sigma": 14}),
    ("median", {"ksize": 5}),
    ("resize", {"scale": 0.6}),
]

def do_attack(img, name, params):
    """Dispatch to attack functions defined in attacks.py."""
    if name == "jpeg":   return A.attack_jpeg(img, **params)
    if name == "awgn":   return A.attack_awgn(img, **params)
    if name == "blur":   return A.attack_blur(img, **params)
    if name == "median": return A.attack_median(img, **params)
    if name == "resize": return A.attack_resize(img, **params)
    if name == "sharp":  return A.attack_sharp(img, **params)
    return img

# ----------------------------
# ROC data collection
# ----------------------------
scores, labels = [], []
counter = 0

for path in IMAGES:
    counter += 1
    orig = cv2.imread(path, 0)
    if orig is None:
        print(f"Warning: could not read {path}, skipping.")
        continue

    # Per-image fixed coordinate set and derived helpers
    H, W = orig.shape
    locs  = _fixed_coords(H, W, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
    wfreq = _weights_for_locs(locs)
    low_idx, mid_idx, high_idx = _band_masks(locs)

    # Create watermarked image and extract its feature vector
    wm = embedding(path, MARK)
    v_wm = extract_ratio_vector(wm, orig, locs)
    v_wm = _winsorize(v_wm, p=2.0)

    # ---------- Positives (light attacks) ----------
    for name, params in ATT_LIGHT:
        att = do_attack(wm, name, params)
        v_att = extract_ratio_vector(att, orig, locs)
        v_att = _winsorize(v_att, p=2.0)
        # Calibrate attacked feature per band before scoring
        v_att_cal = _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))
        scores.append(similarity_weighted(v_wm, v_att_cal, wfreq))
        labels.append(1)

    # ---------- Negatives & baselines (MUST stay inside the image loop) ----------
    # Hard negatives: strong attacks (winsorize + per-band calibration)
    for name, params in NEG_HARD:
        att_hard = do_attack(wm, name, params)
        v_att_hard = extract_ratio_vector(att_hard, orig, locs)
        v_att_hard = _winsorize(v_att_hard, p=2.0)
        v_att_hard_cal = _calibrate_per_band(v_wm, v_att_hard, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))
        scores.append(similarity_weighted(v_wm, v_att_hard_cal, wfreq))
        labels.append(0)

    # Random-noise image (winsorize only; no calibration)
    noise = np.random.randint(0, 256, orig.shape, dtype=np.uint8)
    v_noise = extract_ratio_vector(noise, orig, locs)
    v_noise = _winsorize(v_noise, p=2.0)
    scores.append(similarity_weighted(v_wm, v_noise, wfreq))
    labels.append(0)

    # Clean original (winsorize only; should score low vs v_wm)
    v_clean = extract_ratio_vector(orig, orig, locs)
    v_clean = _winsorize(v_clean, p=2.0)
    scores.append(similarity_weighted(v_wm, v_clean, wfreq))
    labels.append(0)

    # Pure random vector baseline (winsorize only; same length as v_wm)
    rand = np.random.randn(v_wm.size).astype(np.float32)
    rand = _winsorize(rand, p=2.0)
    scores.append(similarity_weighted(v_wm, rand, wfreq))
    labels.append(0)

    if counter % 10 == 0:
        print(f"Processed {counter}/{len(IMAGES)} images...")

# ----------------------------
# ROC computation + tau export
# ----------------------------
scores = np.array(scores, np.float32)
labels = np.array(labels, np.int32)

fpr, tpr, thr = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Select tau at the best Youden’s J within a target FPR budget
target_fpr = 0.15   # was 0.10; relax if you need more recall on light attacks
valid = np.where(fpr <= target_fpr)[0]
if valid.size:
    j = tpr[valid] - fpr[valid]
    idx = valid[np.argmax(j)]
else:
    idx = np.argmin(fpr)

tau = float(thr[idx])

with open("tau.json", "w") as f:
    json.dump({"tau": tau, "AUC": float(roc_auc), "FPR": float(fpr[idx]), "TPR": float(tpr[idx])}, f, indent=2)

print(f"\nROC completed on {len(IMAGES)} images.")
print(f"AUC={roc_auc:.3f}  tau={tau:.6f}  @ FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}")

# Save the ROC plot (PNG + PDF)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, color='navy', label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],"--", color="gray", lw=1)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (100 Images)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc.png", dpi=300)
plt.savefig("roc.pdf")
plt.close()

# Also write tau as plain text for legacy loaders
with open("threshold.txt", "w") as f:
    f.write(str(tau))

# helper for other modules
def get_threshold():
    return float(tau)
    
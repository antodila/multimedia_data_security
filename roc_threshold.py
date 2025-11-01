import json, glob, os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc  # For ROC analysis
import matplotlib
matplotlib.use("Agg")  # Use 'Agg' backend: non-interactive, saves plots directly to file
import matplotlib.pyplot as plt
from embedding import embedding  # Import the embedding function
from scipy.fft import dct
import attacks as A  # Import the attack functions module

# -------------------
# Config
# -------------------
IMG_FOLDER = "images"  # Directory containing the original source images
MARK_SIZE  = 1024      # Watermark length (must match embedding/detection)
SEED       = 123       # Global seed for reproducible randomness
MID_LO, MID_HI = 1, 100  # Frequency band (u+v) for coefficient selection

# Target FPR per scelta soglia
TARGET_FPR = 0.05  # Our desired maximum False Positive Rate for threshold selection

# Find all .bmp images in the folder
IMAGES = sorted(glob.glob(os.path.join(IMG_FOLDER, "0*.bmp")))
print(f"Found {len(IMAGES)} images for ROC computation.")

# -------------------
# Helper DCT/feature
# (These are helper functions for feature extraction,
# matching the logic in the detection module)
# -------------------
def _prefilter(img):
    """Apply a very light Gaussian blur to stabilize DCT coefficients."""
    return cv2.GaussianBlur(img, (3,3), 0.5)

def _2d_dct(x):
    """Computes the 2D Discrete Cosine Transform (orthonormal version)."""
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    """Get all (u,v) coordinates where the frequency sum is in [lo, hi]."""
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """Deterministically select k coordinates from the midband using a fixed seed."""
    rng_local = np.random.default_rng(seed)
    coords = _midband_coords(h, w, lo, hi)
    # Expand the frequency band if not enough coordinates are found
    while len(coords) < k and hi < (h + w - 2):
        hi += 2
        coords = _midband_coords(h, w, lo, hi)
    if len(coords) < k:
        k = len(coords)  # Adjust k if still not enough (e.g., for tiny images)
    # Sample k indices without replacement
    idx = rng_local.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

def _winsorize(x, p=2.0):
    """Clip vector tails at the p-th and (100-p)-th percentiles for robustness."""
    lo, hi = np.percentile(x, p), np.percentile(x, 100.0 - p)
    return np.clip(x, lo, hi)

def _band_masks(locs):
    """Get index masks to separate coordinates into low/mid/high frequency bands."""
    low = []; mid = []; high = []
    for idx, (u,v) in enumerate(locs):
        s = u+v
        if   s <= 15: low.append(idx)
        elif s <= 50: mid.append(idx)
        else:         high.append(idx)
    return np.array(low), np.array(mid), np.array(high)

def _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0)):
    """
    Per-band linear gain calibration.
    Scales v_att to best match v_wm in each frequency band.
    """
    eps = 1e-9
    def gain(a, b):
        """Calculate optimal gain g = argmin ||a - g*b||^2, which is <a,b> / <b,b>."""
        num = float(np.dot(a, b))
        den = float(np.dot(b, b)) + eps
        g = num / den if den > 0 else 1.0
        return float(np.clip(g, clip[0], clip[1]))  # Clamp gain to prevent overfitting
    
    v_att_cal = v_att.copy()
    # Apply gain scaling to each band independently
    for idxs in (low_idx, mid_idx, high_idx):
        if idxs.size == 0:
            continue
        g = gain(v_wm[idxs], v_att[idxs])
        v_att_cal[idxs] = g * v_att[idxs]
    return v_att_cal

def _weights_for_locs(locs):
    """Define frequency-dependent weights for the similarity calculation."""
    w = []
    for (u,v) in locs:
        s = u+v
        if   s <= 15: w.append(1.50)  # High weight for stable low frequencies
        elif s <= 50: w.append(1.10)  # Medium weight
        else:         w.append(0.40)  # Low weight for volatile high frequencies
    return np.asarray(w, dtype=np.float32)

def extract_ratio_vector(img, orig, locs):
    """
    Feature extraction: R(img) = C(prefilter(img)) / C(orig) - 1
    Calculates the relative change in DCT coefficients at 'locs'.
    """
    C_img  = _2d_dct(_prefilter(img).astype(np.float32)) # Prefilter the test image
    C_orig = _2d_dct(orig.astype(np.float32))          # Use the clean original as reference
    eps = 1e-6
    # Create a 'safe' denominator to avoid division by zero
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig)*eps, C_orig)
    # Calculate the ratio at each specified location
    return np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs], dtype=np.float32)

def similarity_weighted(a, b, w):
    """Computes Z-score + weighted cosine similarity between two vectors."""
    n = min(a.size, b.size, w.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32); w = w[:n]
    # Z-score standardization (mean 0, std 1)
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    # Weighted dot product
    num = float(np.sum(w * a * b))
    # Weighted norms
    den = np.sqrt(float(np.sum(w * a*a))) * np.sqrt(float(np.sum(w * b*b))) + 1e-12
    return num / den

def do_attack(img, name, params):
    """Simple dispatcher to call a specific attack function from the 'attacks' module."""
    if name == "jpeg":   return A.attack_jpeg(img, **params)
    if name == "awgn":   return A.attack_awgn(img, **params)
    if name == "blur":   return A.attack_blur(img, **params)
    if name == "median": return A.attack_median(img, **params)
    if name == "resize": return A.attack_resize(img, **params)
    if name == "sharp":  return A.attack_sharp(img, **params)
    return img  # Return original if attack name is not recognized

# Attacchi leggeri per i POSITIVI
# Define a list of light attacks to generate True Positive samples
ATT_LIGHT = [
    ("jpeg",   {"qf": 90}),     # Very high quality JPEG
    ("blur",   {"ksize": 3}),    # Very mild blur
    ("resize", {"scale": 0.9}), # Very light downscaling
]

# -------------------
# Costruzione ROC (ROC Construction)
# -------------------
scores, labels = [], []  # Lists to store all detection scores and their true labels (0 or 1)
counter = 0

# Genera UNA chiave watermark per tutta la ROC run (come nel test del prof)
# Generate ONE global watermark key for the entire ROC run.
# This simulates the real-world scenario where one key is tested against many images.
rng_top = np.random.default_rng(SEED)
wm_bits_global = rng_top.integers(0, 2, size=MARK_SIZE, endpoint=False).astype(np.uint8)
# Save this key to a temporary file for the 'embedding' function to read
np.save("tmp_roc_wm.npy", wm_bits_global)

# Loop over all found images to generate scores
for path in IMAGES:
    counter += 1
    orig = cv2.imread(path, 0)  # Read image in grayscale
    if orig is None:
        print(f"Warning: could not read {path}, skipping.")
        continue

    # --- Pre-computation for this image ---
    H, W = orig.shape
    # Get the fixed set of coordinates, weights, and band masks
    locs  = _fixed_coords(H, W, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
    wfreq = _weights_for_locs(locs)
    low_idx, mid_idx, high_idx = _band_masks(locs)

    # === genera un watermark RANDOM e incorpora (come farà il prof) ===
    # Embed the global watermark into the current original image
    wm_img = embedding(path, "tmp_roc_wm.npy")

    # --- Extract reference feature vectors ---
    # Feature from the (unattacked) watermarked image
    v_wm    = extract_ratio_vector(wm_img, orig, locs)
    # Feature from the clean, original image (this represents the baseline "noise")
    v_clean = extract_ratio_vector(orig,   orig, locs)

    # Stabilize features by clipping extreme tails
    v_wm  = _winsorize(v_wm, p=2.0)
    v_clean = _winsorize(v_clean, p=2.0)

    # Center the watermark feature vector by subtracting the baseline noise.
    # This isolates the watermark signal itself.
    v_wm_c = v_wm - v_clean

    # ---- POSITIVI: attacchi leggeri sull’immagine watermarked ----
    # Generate TRUE POSITIVE samples (label=1)
    for name, params in ATT_LIGHT:
        # Apply one of the light attacks to the watermarked image
        att = do_attack(wm_img, name, params)
        # Extract features from the attacked image
        v_att = extract_ratio_vector(att, orig, locs)
        v_att = _winsorize(v_att, p=2.0)
        # Center the attacked vector (by subtracting the same baseline)
        v_att_c = v_att - v_clean
        # Calibrate the centered, attacked vector against the reference watermark vector
        v_att_c = _calibrate_per_band(v_wm_c, v_att_c, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))
        # Calculate similarity and store the score and label
        scores.append(similarity_weighted(v_wm_c, v_att_c, wfreq))
        labels.append(1)

    # ---- NEGATIVI: detection sull’ORIGINALE (come nel test del prof) ----
    # Generate TRUE NEGATIVE samples (label=0)
    # This simulates running detection on an unwatermarked (original) image
    # The 'attacked' vector is just the feature vector from the original image itself
    v_att = extract_ratio_vector(orig, orig, locs) # Note: v_att is from 'orig'
    v_att = _winsorize(v_att, p=2.0)
    # Centering this vector results in (v_clean - v_clean), which is near-zero
    v_att_c = v_att - v_clean
    # Calibrate this near-zero vector against the reference watermark
    v_att_c = _calibrate_per_band(v_wm_c, v_att_c, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))
    # The similarity score should be low
    scores.append(similarity_weighted(v_wm_c, v_att_c, wfreq))
    labels.append(0)

    if counter % 10 == 0:
        print(f"Processed {counter}/{len(IMAGES)} images...")

# --- ROC Calculation ---
# Convert lists to numpy arrays for sklearn
scores = np.array(scores, np.float32)
labels = np.array(labels, np.int32)

# Compute ROC curve (FPRs, TPRs, and the thresholds that produce them)
fpr, tpr, thr = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)  # Calculate the Area Under the Curve

# --- Threshold Selection ---
# Find all points on the curve where FPR is at or below our target
valid = np.where(fpr <= TARGET_FPR)[0]
if valid.size:
    # From these valid points, pick the one that maximizes (TPR - FPR)
    # This is Youden's J statistic, a common way to select the "best" threshold
    j = tpr[valid] - fpr[valid]
    idx = valid[np.argmax(j)]
else:
    # Fallback: if no point meets the FPR target, just pick the point with the lowest FPR
    idx = np.argmin(fpr)

# This is our chosen detection threshold
tau = float(thr[idx])

# --- Save Results ---
# Save a JSON file with detailed metrics
with open("tau.json", "w") as f:
    json.dump({"tau": tau, "AUC": float(roc_auc), "FPR": float(fpr[idx]), "TPR": float(tpr[idx])}, f, indent=2)

print(f"\nROC completed on {len(IMAGES)} images.")
print(f"AUC={roc_auc:.3f}  tau={tau:.6f}  @ FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}")

# Plot the ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, color='navy', label=f"AUC={roc_auc:.3f}")  # The ROC curve
plt.plot([0,1],[0,1],"--", color="gray", lw=1)  # The 45-degree random guess line
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (centered features)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save the plot in two formats
plt.savefig("roc.png", dpi=300)
plt.savefig("roc.pdf")
plt.close()  # Close the plot to free up memory

# Save a simple text file with *only* the threshold
# This is likely read by the main detection script
with open("threshold.txt", "w") as f:
    f.write(str(tau))

def get_threshold():
    """A simple helper function to return the computed threshold."""
    return float(tau)
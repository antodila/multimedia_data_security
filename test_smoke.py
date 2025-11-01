import cv2
import numpy as np

# Import the embedding function
from embedding import embedding
# Import specific components from the detection module
# This script re-implements the core detection logic to analyze intermediate scores
from detection_shadowmark import (
    _extract_ratio_vector, _fixed_coords, _winsorize,
    _band_masks, _calibrate_per_band, wpsnr,
    SEED, MARK_SIZE, MID_LO, MID_HI,  # Import constants
)
# Import the module containing all attack functions
import attacks as A

# --- Constants for this test script ---
IMG  = "lena_grey.bmp"   # The single source image to test
MARK = "shadowmark.npy" # The watermark key file to embed

def _weights_for_locs(locs):
    """
    Helper function (local copy):
    Define frequency-dependent weights for the similarity calculation.
    """
    w = []
    for (u, v) in locs:
        s = u + v
        if s <= 15:   w.append(1.50)  # low frequencies (high weight)
        elif s <= 50: w.append(1.10)  # mid frequencies (medium weight)
        else:         w.append(0.40)  # high frequencies (low weight)
    return np.asarray(w, dtype=np.float32)

def similarity_weighted(a, b, w):
    """
    Helper function (local copy):
    Computes Z-score + weighted cosine similarity between two vectors.
    """
    n = min(a.size, b.size, w.size)
    a = a[:n].astype(np.float32)
    b = b[:n].astype(np.float32)
    w = w[:n]
    
    # z-score standardization (mean 0, std 1)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    
    # Calculate weighted cosine similarity
    num = float(np.sum(w * a * b))
    den = (np.sqrt(float(np.sum(w * a * a))) *
           np.sqrt(float(np.sum(w * b * b))) + 1e-12)
    return num / den

# --- 1. Load images and prepare ---

# Load the original image in grayscale
orig = cv2.imread(IMG, 0)
if orig is None:
    raise FileNotFoundError(f"Missing {IMG}")

# Embed the watermark into the original image
wm_img = embedding(IMG, MARK)

# --- 2. Prepare for feature extraction ---
# Get image dimensions
H, W = orig.shape
# Get the fixed, deterministic set of DCT coordinates to sample
locs  = _fixed_coords(H, W, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
# Get the frequency weights corresponding to those coordinates
wfreq = _weights_for_locs(locs)
# Get the index masks for low/mid/high frequency bands (for calibration)
low_idx, mid_idx, high_idx = _band_masks(locs)

# --- 3. Extract baseline feature vectors ---
# Note: _extract_ratio_vector contains the prefiltering step
# Get the "ground truth" feature vector from the unattacked watermarked image
v_wm    = _extract_ratio_vector(wm_img, orig, locs)
# Stabilize the vector by clipping extreme tails
v_wm    = _winsorize(v_wm, p=2.0)

# Get the "baseline" feature vector from the original (clean) image
v_clean = _extract_ratio_vector(orig, orig, locs)
v_clean = _winsorize(v_clean, p=2.0)

# --- 4. Sanity checks ---
# Check similarity of the watermark vector with itself (should be 1.0)
print("TP self-sim (≈1):", similarity_weighted(v_wm, v_wm, wfreq))
# Check similarity of the watermark vector with the clean vector (should be low/near 0)
print("TN sim (wm vs clean):", similarity_weighted(v_wm, v_clean, wfreq))

# --- 5. Attacks Battery ---
# Define a list of attacks and their parameters to test
candidates = [
    ("jpeg",   {"qf": 70}),
    ("awgn",   {"sigma": 6}),
    ("blur",   {"ksize": 5}),
    ("median", {"ksize": 3}),
    ("resize", {"scale": 0.6}),
]

# Loop through each defined attack
for name, params in candidates:
    
    # Apply the current attack to the watermarked image
    if name == "jpeg":   att = A.attack_jpeg(wm_img, **params)
    if name == "awgn":   att = A.attack_awgn(wm_img, **params)
    if name == "blur":   att = A.attack_blur(wm_img, **params)
    if name == "median": att = A.attack_median(wm_img, **params)
    if name == "resize": att = A.attack_resize(wm_img, **params)

    # --- 6. Analyze the attacked image ---
    
    # Extract the feature vector from the attacked image
    v_att = _extract_ratio_vector(att, orig, locs)
    # Stabilize the attacked vector
    v_att = _winsorize(v_att, p=2.0)
    
    # Apply per-band gain calibration:
    # This crucial step scales the attacked vector (v_att) to match the
    # energy of the ground truth vector (v_wm) in each band,
    # compensating for gain changes (like from compression).
    v_att_cal = _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))

    # Calculate the final detection score:
    # Similarity between the ground truth (v_wm) and the calibrated attacked vector (v_att_cal)
    sim_w = similarity_weighted(v_wm, v_att_cal, wfreq)

    # --- 7. Calculate quality metrics ---
    # WPSNR vs watermarked: How much did the attack change the watermarked image? (Robustness)
    w_wm = wpsnr(wm_img, att)
    # WPSNR vs original: How close is the attacked image to the *original*? (Fidelity)
    w_or = wpsnr(orig,   att)

    # Print the results for this attack
    print(f"{name} {params} -> sim={sim_w:.3f}, WPSNR(wm,att)={w_wm:.2f}, WPSNR(orig,att)={w_or:.2f}")
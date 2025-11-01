import os, glob, json
import numpy as np
import cv2

# Import the detector's internal building blocks so we can reproduce
# the exact same feature extraction + scoring pipeline used at detection time.
# This is crucial: threshold tuning must mirror the detector perfectly.
from detection_shadowmark import (
    _fixed_coords, _weights_for_locs, _band_masks,
    _extract_ratio_vector, _winsorize, _calibrate_per_band,
    _similarity_weighted, SEED, MARK_SIZE, MID_LO, MID_HI
)
from embedding import embedding  # Watermark embedding function

# --- Configuration ---
IMG_FOLDER     = "images"                 # Folder containing your ~101 *.bmp images
PATTERN        = os.path.join(IMG_FOLDER, "0*.bmp")  # Glob pattern to locate images
M_RANDOM_WM    = 5                        # Per professor's request: at least 5 different random WMs
TAU_TARGET_FPR = 0.05                     # Target FPR = 5% for threshold selection

def main():
    # Collect candidate images; we need a sufficiently large set for statistical reliability.
    images = sorted(glob.glob(PATTERN))
    assert len(images) >= 50, "Dataset is too small: need ~100 test images."

    # RNG used ONLY to generate random watermark bits (distinct from any detector RNG).
    rng = np.random.default_rng(SEED + 999)

    # We'll accumulate "negative" scores here.
    # A "negative" is the score obtained when comparing a watermark to a CLEAN image,
    # i.e., the situation where the watermark should *not* be detected.
    neg_scores = []

    # Outer loop over multiple random watermarks to avoid overfitting to a single key.
    for m in range(M_RANDOM_WM):

        # Generate a new 1024-bit random watermark for this iteration.
        # Using dtype=uint8 keeps the embedding fast and memory-light.
        W = rng.integers(0, 2, size=MARK_SIZE, dtype=np.uint8)

        # Save to disk because the embedding() API expects a .npy watermark path.
        np.save("tmp_watermark.npy", W)

        # Iterate over the whole clean dataset for this particular watermark.
        for img_path in images:
            # Read the original (clean) image in grayscale.
            orig = cv2.imread(img_path, 0)

            # Create the watermarked version of the same original using the current random key.
            # This simulates the "reference" watermarked image the detector uses for comparison.
            wm_img = embedding(img_path, "tmp_watermark.npy")

            # Prepare detector parameters exactly as in detection:
            # - the fixed coordinate set in the DCT domain
            # - frequency weights
            # - band masks for per-band calibration
            H, Wimg = orig.shape
            locs   = _fixed_coords(H, Wimg, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
            wfreq  = _weights_for_locs(locs)
            low_idx, mid_idx, high_idx = _band_masks(locs)

            # --- Feature extraction (must match the detector path) ---
            # v_wm:  ratio features from the watermarked image vs the clean reference spectrum
            # v_clean: ratio features from the clean image vs the clean reference spectrum
            v_wm    = _extract_ratio_vector(wm_img, orig, locs)
            v_clean = _extract_ratio_vector(orig,   orig, locs)

            # Winsorize both to stabilize heavy tails (as in detection).
            v_wm    = _winsorize(v_wm, p=2.0)
            v_clean = _winsorize(v_clean, p=2.0)

            # Center both w.r.t. the clean baseline to isolate the watermark component.
            v_wm_c    = v_wm   - v_clean
            v_clean_c = v_clean - v_clean  # this becomes ~zero; we keep it for symmetry

            # Per-band gain calibration (applied to the "attacked" vector in detection).
            # Here the "attacked" stand-in is v_clean_c, calibrated against v_wm_c.
            # The clip range prevents overfitting/unstable gains.
            v_clean_cal = _calibrate_per_band(
                v_wm_c, v_clean_c, low_idx, mid_idx, high_idx, clip=(0.5, 2.0)
            )

            # Compute the *exact same* weighted similarity score that the detector uses.
            score = _similarity_weighted(v_wm_c, v_clean_cal, wfreq)

            # Store this negative-case score (should ideally be small).
            neg_scores.append(float(score))

    # --- Threshold computation ---

    # Convert to NumPy for vectorized stats.
    neg_scores = np.asarray(neg_scores, dtype=np.float32)

    # We pick τ as the (1 - FPR_target) quantile of negative scores.
    # That is: the 95th percentile → ~5% of negatives would exceed τ in the worst case.
    tau_q = float(np.quantile(neg_scores, 1.0 - TAU_TARGET_FPR))

    # Guard against the degenerate case where all negatives are ~0
    # (common when features are strongly centered): enforce a tiny positive floor.
    EPS = 1e-6
    tau = max(tau_q, EPS)

    # Save τ in a JSON file (consistent with the rest of the pipeline).
    with open("tau.json", "w") as f:
        # AUC/TPR are not computed here (this script is FPR-oriented),
        # so we store None for clarity.
        json.dump({"tau": tau, "AUC": None, "FPR": TAU_TARGET_FPR, "TPR": None}, f, indent=2)

    # Report the empirical FPR measured with *strict* inequality (>).
    # Using ">" avoids artificially counting exact-equality ties as false positives
    # when the quantile lands exactly on many identical scores (e.g., all zeros).
    over = np.mean(neg_scores > tau) * 100.0

    print(f"Negatives collected: {len(neg_scores)}")
    print(f"Chosen tau @ {int(TAU_TARGET_FPR*100)}% target FPR: {tau:.6f}")
    print(f"Empirical FPR on clean-set (with {M_RANDOM_WM} random WMs): {over:.2f}%")

if __name__ == "__main__":
    main()

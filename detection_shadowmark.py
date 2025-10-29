import json
import os
import numpy as np
import cv2
from scipy.fft import dct
from scipy.signal import convolve2d

"""
Shadowmark — Detection module (commented)

Key ideas:
- Feature: ratio in DCT domain R(img) = C(img) / C(orig) - 1, sampled at fixed mid-band coordinates.
- Light prefiltering before DCT to stabilize against mild JPEG/resize artifacts.
- Winsorization to cap heavy tails in the feature vector.
- Per-band (low/mid/high) linear gain calibration of the attacked feature to reduce benign distortions.
- Weighted cosine similarity + z-scoring as decision statistic.
- Random-baseline safety check: require similarity to exceed the best random alignment by a small margin.
- Threshold tau is loaded from tau.json/threshold.txt with a safe fallback.

API:
    detection(input1, input2, input3) -> (presence:int, wpsnr:float)
"""

# ========== PARAMETERS ==========
TAU_DEFAULT   = 0.397794      # fallback threshold (will be overridden by tau.json if present)
WPSNR_REJECT  = 25.0          # immediate reject for very degraded images (per challenge rules)

# Previously: RND_TRIALS=30 and MIN_MARGIN=0.05
# We use fewer trials and a smaller margin to avoid over-penalizing true positives
RND_TRIALS    = 10            # number of random watermark trials for the safety check
MIN_MARGIN    = 0.002         # similarity must exceed max(random) by at least this margin

SEED          = 123
MARK_SIZE     = 1024
MID_LO, MID_HI = 1, 100       # band of (u+v) indices for mid-band sampling
# =================================


def _prefilter(img):
    """
    Light denoising before DCT:
    Reduces small JPEG ringing / resize jitter without killing the watermark.
    """
    return cv2.GaussianBlur(img, (3, 3), 0.5)


def _get_locs(h, w, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Convenience wrapper to obtain the fixed mid-band coordinates for sampling.
    """
    return _fixed_coords(h, w, k, seed, lo, hi)


def _weights_for_locs(locs):
    """
    Frequency-dependent weights for weighted cosine similarity.
    Slightly up-weights low/mid bands (more watermark energy and stability),
    down-weights high bands (more vulnerable to attacks).
    """
    w = []
    for (u, v) in locs:
        s = u + v
        if s <= 15:      w.append(1.50)  # was 1.30
        elif s <= 50:    w.append(1.10)  # was 1.00
        else:            w.append(0.40)  # was 0.60
    return np.asarray(w, dtype=np.float32)


def _similarity_weighted(a, b, w):
    """
    Weighted cosine similarity after z-score standardization.

    Args:
        a, b: 1D feature vectors (same length)
        w:    per-coordinate non-negative weights (same length)

    Returns:
        Scalar in [-1, 1], higher = more similar.
    """
    n = min(a.size, b.size, w.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32); w = w[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    num = float(np.sum(w * a * b))
    den = np.sqrt(float(np.sum(w * a * a))) * np.sqrt(float(np.sum(w * b * b))) + 1e-12
    return num / den


def _load_tau():
    """
    Load tau from tau.json (produced by roc_threshold.py), otherwise threshold.txt, otherwise fallback.
    """
    try:
        if os.path.exists("tau.json"):
            with open("tau.json", "r") as f:
                data = json.load(f)
            if "tau" in data:
                return float(data["tau"])
    except Exception:
        pass
    try:
        if os.path.exists("threshold.txt"):
            with open("threshold.txt", "r") as f:
                return float(f.read().strip())
    except Exception:
        pass
    return float(TAU_DEFAULT)


tau = _load_tau()


def _2d_dct(x):
    """Orthogonal 2D-DCT."""
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')


def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    """
    All coordinates whose index-sum lies in [lo, hi], i.e. lo <= (u+v) <= hi.
    """
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]


def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Deterministic random subset of mid-band coordinates of size k.
    Expands 'hi' if needed to ensure enough samples are available.
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


def _extract_ratio_vector(img, orig, locs, mark_size=MARK_SIZE):
    """
    Feature extraction:
        R(img) = C(img) / C(orig) - 1
    computed at the provided frequency coordinates 'locs'.

    Notes:
    - We apply a light Gaussian prefilter to 'img' before DCT to stabilize the ratio.
    - 'orig' goes unfiltered to preserve the clean reference spectrum.
    """
    img_f  = _prefilter(img).astype(np.float32)
    orig_f = orig.astype(np.float32)

    C_img  = _2d_dct(img_f)
    C_orig = _2d_dct(orig_f)

    eps = 1e-6
    # Avoid division by zero on very small coefficients
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig) * eps, C_orig)

    ratio = np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs],
                     dtype=np.float32)
    return ratio


def _winsorize(x, p=2.0):
    """
    Winsorize vector x to clamp extreme tails at [p, 100-p] percentiles.
    Helps robustness against localized artifacts.
    """
    lo, hi = np.percentile(x, p), np.percentile(x, 100.0 - p)
    return np.clip(x, lo, hi)


def _band_masks(locs):
    """
    Build index masks for low/mid/high sub-bands, using (u+v) thresholds.
    Useful for per-band calibration.
    """
    low = []; mid = []; high = []
    for idx, (u, v) in enumerate(locs):
        s = u + v
        if   s <= 15: low.append(idx)
        elif s <= 50: mid.append(idx)
        else:         high.append(idx)
    return np.array(low), np.array(mid), np.array(high)


def _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0)):
    """
    Per-band linear gain calibration of 'v_att' towards 'v_wm':
    For each band B, estimate g_B = argmin || v_wm[B] - g * v_att[B] ||^2
    and clamp g_B to [clip[0], clip[1]] to avoid overfitting.
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


def _similarity(a, b):
    """
    Unweighted cosine similarity after z-score (kept for the random-baseline check).
    """
    n = min(a.size, b.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32)
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    return float(np.sum(a * b) / n)


# --- WPSNR (with CSF), unchanged from your original implementation ---
def _csffun(u, v):
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    Sa = Sw * Ow
    return Sa


def _csfmat():
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    Fmat = _csffun(u, v)
    return Fmat


def _get_csf_filter():
    Fmat = _csfmat()
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    """
    CSF-weighted PSNR. Returns a very large value if images are identical.
    """
    if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
        raise TypeError("Input images must be NumPy arrays.")
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")

    if image_a.dtype not in (np.float64, np.float32):
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()

    if image_b.dtype not in (np.float64, np.float32):
        B = image_b.astype(np.float64) / 255.0
    else:
        B = image_b.copy()

    if (A.max() > 1.0 or A.min() < 0.0) or (B.max() > 1.0 or B.min() < 0.0):
        raise ValueError("Float images must be in [0,1]; uint images in [0,255].")

    if np.array_equal(A, B):
        return 9999999.0

    error_image = A - B
    csf_filter = _get_csf_filter()
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    wmse = np.mean(weighted_error**2)
    if wmse == 0:
        return 9999999.0
    decibels = 20 * np.log10(1.0 / np.sqrt(wmse))
    return decibels
# --- end WPSNR ---


def detection(input1, input2, input3):
    """
    Detection entry point.

    Args:
        input1: path to original (clean) image
        input2: path to watermarked image
        input3: path to attacked/test image

    Returns:
        (presence:int, wpsnr:float)
            presence = 1 if watermark is detected, else 0
            wpsnr    = CSF-weighted PSNR between 'watermarked' and 'attacked'
    """
    # Load images in grayscale
    orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if orig is None or wm is None or att is None:
        raise FileNotFoundError("One or more images not found.")

    # 1) Early reject for severely degraded images (challenge requirement)
    wps = wpsnr(wm, att)
    if wps < WPSNR_REJECT:
        return 0, float(wps)

    # 2) Use the same sampling coordinates and weights for all three images
    H, W = wm.shape
    locs  = _fixed_coords(H, W, k=MARK_SIZE, seed=SEED, lo=MID_LO, hi=MID_HI)
    wfreq = _weights_for_locs(locs)
    low_idx, mid_idx, high_idx = _band_masks(locs)

    # 3) Extract features with prefiltering + winsorization + per-band calibration
    v_wm  = _extract_ratio_vector(wm,  orig, locs)
    v_att = _extract_ratio_vector(att, orig, locs)

    v_wm  = _winsorize(v_wm,  p=2.0)
    v_att = _winsorize(v_att, p=2.0)

    # Calibrate attacked feature per band to reduce benign frequency-dependent drifts
    v_att_cal = _calibrate_per_band(v_wm, v_att, low_idx, mid_idx, high_idx, clip=(0.5, 2.0))

    # 4) Weighted cosine similarity on z-scored features
    sim = _similarity_weighted(v_wm, v_att_cal, wfreq)

    # 5) Random-baseline safety check:
    #    ensure the observed similarity is not explainable by chance alignment
    rng = np.random.default_rng(SEED)
    rnd_sims = []
    for _ in range(RND_TRIALS):
        rnd = rng.normal(size=v_att.shape).astype(np.float32)
        rnd = (rnd - rnd.mean()) / (rnd.std() + 1e-9)
        rnd_sims.append(_similarity(rnd, v_att))
    max_rnd = max(rnd_sims) if rnd_sims else 0.0

    # Final decision: above tau AND above the random baseline by a small margin
    presence = 1 if (sim >= tau and sim > max_rnd + MIN_MARGIN) else 0
    return int(presence), float(wps)

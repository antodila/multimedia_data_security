# detection_group.py
import numpy as np
import cv2
from scipy.fft import dct

# === Set this after running roc_threshold.py ===
#tau = 0.5  # <-- replace with numeric ROC result
tau = 0.307841
# Keep these consistent with embed_spread_spectrum.py
SEED = 123
MID_LO, MID_HI = 4, 60
MARK_SIZE = 1024

def _2d_dct(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    rng_local = np.random.default_rng(seed)
    coords = _midband_coords(h, w, lo, hi)
    # expand band if needed
    while len(coords) < k and hi < (h + w - 2):
        hi += 2
        coords = _midband_coords(h, w, lo, hi)
    if len(coords) < k:
        k = len(coords)
    idx = rng_local.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

def _extract_ratio_vector(img, orig, mark_size=MARK_SIZE):
    """
    Normalized residual in the DCT domain:
    R = C(img)/C(orig) - 1, sampled at the embedding coordinates.
    Cancels image content; isolates multiplicative watermark.
    """
    C_img  = _2d_dct(img.astype(np.float32))
    C_orig = _2d_dct(orig.astype(np.float32))

    # avoid division by ~0 on tiny coefficients
    eps = 1e-6
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig)*eps, C_orig)

    locs = _fixed_coords(*C_img.shape, k=mark_size, seed=SEED)
    ratio = np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs],
                     dtype=np.float32)
    return ratio

def _similarity(a, b):
    n = min(a.size, b.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.sum(a * b) / n)

def wpsnr(img1, img2):
    # Replace with lab WPSNR for the competition (this is PSNR fallback)
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def detection(input1, input2, input3):
    """
    input1: original clean image path
    input2: watermarked image path
    input3: attacked image path
    returns: (presence_int, wpsnr_float)
    """
    orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if orig is None or wm is None or att is None:
        raise FileNotFoundError("One or more images not found")

    v_wm  = _extract_ratio_vector(wm, orig)
    v_att = _extract_ratio_vector(att, orig)

    sim = _similarity(v_wm, v_att)
    wps = wpsnr(wm, att)

    presence = 1 if sim >= tau else 0
    return int(presence), float(wps)

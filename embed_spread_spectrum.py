# embedding.py
import numpy as np
import cv2
from scipy.fft import dct, idct

ALPHA = 0.08
MID_LO, MID_HI = 4, 60   # widen the band (was too narrow for 1024 picks)
SEED = 123               # keep your seed
MARK_SIZE = 1024         # keep 1024 per challenge

rng = np.random.default_rng(SEED)

def _load_watermark(input2, mark_size=MARK_SIZE):
    if isinstance(input2, str):
        W = np.load(input2)
    else:
        W = np.array(input2)
    W = (W.flatten() % 2).astype(np.uint8)
    if W.size < mark_size:
        W = np.tile(W, int(np.ceil(mark_size / W.size)))[:mark_size]
    else:
        W = W[:mark_size]
    return W

def _2d_dct(x):  return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')
def _2d_idct(X): return idct(idct(X, axis=1, norm='ortho'), axis=0, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Deterministically select k mid-band coordinates.
    If the band is still too small, automatically expand it until we have >= k.
    """
    rng_local = np.random.default_rng(seed)
    coords = _midband_coords(h, w, lo, hi)

    # If not enough coords, expand the band upward until we have >= k or we hit the max possible
    while len(coords) < k and hi < (h + w - 2):
        hi += 2
        coords = _midband_coords(h, w, lo, hi)

    if len(coords) < k:
        # fallback: take all available if we still don't have k
        k = len(coords)

    idx = rng_local.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

def embedding(input1, input2):
    """
    input1: path to grayscale BMP 512x512
    input2: path to .npy watermark or array-like
    return : watermarked image (uint8), no prints, no saving
    """
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")
    imgf = img.astype(np.float32)

    W = _load_watermark(input2, MARK_SIZE)
    S = (2.0 * W.astype(np.float32) - 1.0)  # map {0,1} -> {-1,+1}

    C = _2d_dct(imgf)
    sign = np.sign(C); Cabs = np.abs(C)

    locs = _fixed_coords(*C.shape, k=MARK_SIZE, seed=SEED)

    Cmod = Cabs.copy()
    for i, (u, v) in enumerate(locs):
        Cmod[u, v] *= (1.0 + ALPHA * S[i])

    Cmod *= sign
    out = _2d_idct(Cmod)
    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    return out

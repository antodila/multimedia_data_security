# detection_shadowmark.py
# Group: shadowmark
# Detection with residual ratio in DCT domain. Self-contained.
# Includes professor's WPSNR implementation (inline). No prints.

import numpy as np
import cv2
from scipy.fft import dct
from scipy.signal import convolve2d

# ---- Threshold from your ROC run ----
tau=0.031963  # Updated optimal threshold from ROC computation

# ---- Must mirror embedding constants ----
SEED = 123
MARK_SIZE = 1024
MID_LO, MID_HI = 1, 100

# ---- DCT helpers ----
def _2d_dct(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

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

# ---- Feature: residual ratio vector (cancels image content) ----
def _extract_ratio_vector(img, orig, mark_size=MARK_SIZE):
    """
    R(img) = C(img)/C(orig) - 1  over the embedding coordinates.
    """
    C_img  = _2d_dct(img.astype(np.float32))
    C_orig = _2d_dct(orig.astype(np.float32))

    eps = 1e-6
    C_orig_safe = np.where(np.abs(C_orig) < eps, np.sign(C_orig)*eps, C_orig)

    locs = _fixed_coords(*C_img.shape, k=mark_size, seed=SEED)
    ratio = np.array([(C_img[u, v] / C_orig_safe[u, v]) - 1.0 for (u, v) in locs],
                     dtype=np.float32)
    return ratio

def _similarity(a, b):
    n = min(a.size, b.size)
    a = a[:n].astype(np.float32); b = b[:n].astype(np.float32)
    a = (a - a.mean())/(a.std() + 1e-9)
    b = (b - b.mean())/(b.std() + 1e-9)
    return float(np.sum(a*b)/n)

# ---- Professor's WPSNR (as provided) ----
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
    if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
        raise TypeError("Input images must be NumPy arrays.")
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")

    if image_a.dtype != np.float64 and image_a.dtype != np.float32:
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()

    if image_b.dtype != np.float64 and image_b.dtype != np.float32:
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

# ---- Public API ----
def detection(input1, input2, input3):
    """
    input1: path to original (clean) image
    input2: path to watermarked image
    input3: path to attacked image
    return: (presence_int, wpsnr_float)
    """
    orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if orig is None or wm is None or att is None:
        raise FileNotFoundError("One or more images not found.")

    v_wm  = _extract_ratio_vector(wm, orig)
    v_att = _extract_ratio_vector(att, orig)

    sim = _similarity(v_wm, v_att)
    wps = wpsnr(wm, att)

    presence = 1 if sim >= tau else 0
    return int(presence), float(wps)
import numpy as np
import cv2
from scipy.fft import dct, idct

# ----------------------------
# Embedding hyperparameters
# ----------------------------
ALPHA = 0.015          # Base modulation strength (scaled adaptively per coeff)
MARK_SIZE = 1024       # Number of DCT coefficients to watermark
SEED = 123             # Fixed seed for reproducibility (coords selection)
MID_LO, MID_HI = 1, 100  # Candidate band: all (u+v) in this range
REDUNDANCY_FACTOR = 1  # >1 would spread each bit to neighbors for extra robustness

def _load_watermark(input2, mark_size=MARK_SIZE):
    """
    Load the watermark as a binary vector of length 'mark_size'.
    - Accepts a .npy path or any array-like.
    - Tiles or truncates to exactly 'mark_size'.
    """
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

# Orthonormal separable 2D-DCT/IDCT
def _2d_dct(x):  return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')
def _2d_idct(X): return idct(idct(X, axis=1, norm='ortho'), axis=0, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    """
    Return all (u,v) coordinates whose taxicab frequency sum u+v lies in [lo, hi].
    This defines the candidate region where we can embed (low+mid+some high).
    """
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Deterministically select 'k' coordinates with a frequency-aware sampling:
    - 25% from low (u+v ≤ 15), 50% from mid (15 < u+v ≤ 50), 25% from high (50 < u+v ≤ 100)
    - If not enough coords are available, top up from the global midband [lo, hi]
    The fixed RNG seed ensures reproducibility across runs.
    """
    rng_local = np.random.default_rng(seed)

    low_coords  = [(u, v) for u in range(h) for v in range(w) if 1 <= (u + v) <= 15]
    mid_coords  = [(u, v) for u in range(h) for v in range(w) if 15 < (u + v) <= 50]
    high_coords = [(u, v) for u in range(h) for v in range(w) if 50 < (u + v) <= 100]
    
    low_count  = min(k // 4, len(low_coords))     # 25% low frequency
    mid_count  = min(k // 2, len(mid_coords))     # 50% mid frequency
    high_count = min(k - low_count - mid_count, len(high_coords))  # 25% high

    selected_coords = []
    if low_count > 0:
        idx = rng_local.choice(len(low_coords), size=low_count, replace=False)
        selected_coords.extend([low_coords[i] for i in idx])
    if mid_count > 0:
        idx = rng_local.choice(len(mid_coords), size=mid_count, replace=False)
        selected_coords.extend([mid_coords[i] for i in idx])
    if high_count > 0:
        idx = rng_local.choice(len(high_coords), size=high_count, replace=False)
        selected_coords.extend([high_coords[i] for i in idx])

    # If we still need more, pick from the generic midband [lo, hi] excluding already chosen
    remaining = k - len(selected_coords)
    if remaining > 0:
        all_coords = _midband_coords(h, w, lo, hi)
        available = [c for c in all_coords if c not in selected_coords]
        if len(available) >= remaining:
            idx = rng_local.choice(len(available), size=remaining, replace=False)
            selected_coords.extend([available[i] for i in idx])

    # Ensure exactly k coords (truncate if any rounding mismatch)
    return selected_coords[:k]

def embedding(input1, input2):
    """
    Embed a spread-spectrum watermark into a grayscale image (DCT domain).
    Args:
      input1: path to the original grayscale BMP (commonly 512x512)
      input2: path to watermark .npy OR an array-like (binary)
    Returns:
      np.uint8 watermarked image (same size as input), without writing to disk.

    Method:
      - Compute 2D-DCT of the image.
      - Select MARK_SIZE coefficients via _fixed_coords (frequency-aware selection).
      - For each selected (u,v), scale its magnitude by (1 + alpha * s_i),
        where s_i ∈ {-1, +1} comes from the watermark.
      - Alpha is adaptively adjusted based on frequency, coefficient magnitude, and position.
      - Re-apply original DCT sign and inverse DCT back to spatial domain.
    """
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")
    imgf = img.astype(np.float32)

    # Load watermark bits: W ∈ {0,1}; map to S ∈ {-1,+1}
    W = _load_watermark(input2, MARK_SIZE)
    S = (2.0 * W.astype(np.float32) - 1.0)  # {0,1} -> {-1,+1}

    # DCT of the image; we modify magnitudes, preserve signs
    C = _2d_dct(imgf)
    sign = np.sign(C)
    Cabs = np.abs(C)

    # Coordinate set where we embed the watermark
    locs = _fixed_coords(*C.shape, k=MARK_SIZE, seed=SEED)

    # Work on a copy of magnitudes (avoid touching phase/sign)
    Cmod = Cabs.copy()
    
    for i, (u, v) in enumerate(locs):
        coeff_magnitude = Cabs[u, v]
        freq_sum = u + v

        # Only modify sufficiently energetic coefficients to avoid artifacts
        if coeff_magnitude > 1.0:
            # Frequency-dependent factor (slightly stronger at low/high than mid)
            if freq_sum <= 15:
                freq_factor = 1.2   # low frequencies: stable & impactful
            elif freq_sum <= 50:
                freq_factor = 1.0   # mid band: baseline
            else:
                freq_factor = 1.1   # high band: moderate boost

            # Magnitude-dependent factor: stronger push when coefficient is large
            magnitude_factor = 1.0 + 0.1 * np.log(1.0 + coeff_magnitude)

            # Position-dependent factor: very mild tapering across the matrix
            u_norm = u / C.shape[0]
            v_norm = v / C.shape[1]
            position_factor = 1.0 - 0.01 * (u_norm + v_norm)

            # Final adaptive alpha per coefficient
            comprehensive_alpha = ALPHA * freq_factor * magnitude_factor * position_factor

            # Apply the modulation:
            # - Primary write on (u,v)
            # - Optional redundant writes around (u,v) if REDUNDANCY_FACTOR > 1
            for redundancy in range(REDUNDANCY_FACTOR):
                if redundancy == 0:
                    Cmod[u, v] *= (1.0 + comprehensive_alpha * S[i])
                else:
                    # Spread a fraction of the same symbol to a small neighborhood
                    offset = redundancy
                    for du in [-offset, 0, offset]:
                        for dv in [-offset, 0, offset]:
                            if du == 0 and dv == 0:
                                continue
                            nu, nv = u + du, v + dv
                            if 0 <= nu < C.shape[0] and 0 <= nv < C.shape[1]:
                                if Cabs[nu, nv] > 0.1:  # ignore near-zero neighbors
                                    Cmod[nu, nv] *= (1.0 + comprehensive_alpha * 0.3 * S[i])

    # Restore original signs and invert DCT back to spatial domain
    Cmod *= sign
    out = _2d_idct(Cmod)

    # Quantize to 8-bit with safe clipping
    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    return out

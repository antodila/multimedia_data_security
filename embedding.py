# embedding.py
# Group: shadowmark
# Spread-spectrum embedding in the 2D-DCT domain (multiplicative),
# deterministic mid-band coordinates. Self-contained, no I/O or prints.

import numpy as np
import cv2
from scipy.fft import dct, idct

# ---- Tunables (must mirror detection constants) ----
ALPHA = 0.015  # ULTRA-FINE-TUNED: Push quality to 50+ dB while maintaining robustness
MARK_SIZE = 1024
SEED = 123
MID_LO, MID_HI = 1, 100  # Maximum frequency band for comprehensive robustness
REDUNDANCY_FACTOR = 1  # Minimal redundancy for maximum quality

# ---- Helpers ----
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
    rng_local = np.random.default_rng(seed)
    
    # MULTI-LAYERED FREQUENCY STRATEGY: Comprehensive coverage for all attack types
    # Low frequency: Robust against compression and noise
    low_coords = [(u, v) for u in range(h) for v in range(w) if 1 <= (u + v) <= 15]
    # Mid frequency: Robust against filtering and moderate attacks  
    mid_coords = [(u, v) for u in range(h) for v in range(w) if 15 < (u + v) <= 50]
    # High frequency: Robust against geometric attacks
    high_coords = [(u, v) for u in range(h) for v in range(w) if 50 < (u + v) <= 100]
    
    # Distribute watermark across all frequency bands (comprehensive robustness)
    low_count = min(k // 4, len(low_coords))  # 25% low frequency
    mid_count = min(k // 2, len(mid_coords))  # 50% mid frequency  
    high_count = min(k - low_count - mid_count, len(high_coords))  # 25% high frequency
    
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
    
    # Fill remaining slots with any available coordinates
    remaining = k - len(selected_coords)
    if remaining > 0:
        all_coords = _midband_coords(h, w, lo, hi)
        available = [c for c in all_coords if c not in selected_coords]
        if len(available) >= remaining:
            idx = rng_local.choice(len(available), size=remaining, replace=False)
            selected_coords.extend([available[i] for i in idx])
    
    return selected_coords[:k]

# ---- Public API ----
def embedding(input1, input2):
    """
    input1: path to original grayscale BMP image (512x512)
    input2: path to watermark (.npy) or array-like (binary)
    return : watermarked image (np.uint8), no side effects
    """
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")
    imgf = img.astype(np.float32)

    W = _load_watermark(input2, MARK_SIZE)
    S = (2.0 * W.astype(np.float32) - 1.0)  # {0,1} -> {-1,+1}

    C = _2d_dct(imgf)
    sign = np.sign(C); Cabs = np.abs(C)

    locs = _fixed_coords(*C.shape, k=MARK_SIZE, seed=SEED)

    Cmod = Cabs.copy()
    
    # COMPREHENSIVE ROBUSTNESS STRATEGY: Multi-layered embedding for all attack types
    for i, (u, v) in enumerate(locs):
        coeff_magnitude = Cabs[u, v]
        freq_sum = u + v
        
        # MAXIMUM QUALITY STRATEGY: Minimal embedding for highest WPSNR
        if coeff_magnitude > 1.0:  # High threshold for maximum quality
            
            # Frequency-based alpha adjustment
            if freq_sum <= 15:  # Low frequency: Maximum robustness against compression/noise
                freq_factor = 1.2
            elif freq_sum <= 50:  # Mid frequency: Balanced robustness
                freq_factor = 1.0
            else:  # High frequency: Strong robustness against geometric attacks
                freq_factor = 1.1
            
            # Magnitude-based factor (minimal embedding for maximum quality)
            magnitude_factor = 1.0 + 0.1 * np.log(1.0 + coeff_magnitude)
            
            # Position-based factor (minimal reduction for maximum robustness)
            u_norm = u / C.shape[0]
            v_norm = v / C.shape[1]
            position_factor = 1.0 - 0.01 * (u_norm + v_norm)  # Minimal reduction
            
            # Comprehensive robust alpha
            comprehensive_alpha = ALPHA * freq_factor * magnitude_factor * position_factor
            
            # Apply embedding with redundancy
            for redundancy in range(REDUNDANCY_FACTOR):
                if redundancy == 0:
                    Cmod[u, v] *= (1.0 + comprehensive_alpha * S[i])
                else:
                    # Additional redundant embedding in nearby coefficients
                    offset = redundancy
                    for du in [-offset, 0, offset]:
                        for dv in [-offset, 0, offset]:
                            if du == 0 and dv == 0:
                                continue
                            nu, nv = u + du, v + dv
                            if 0 <= nu < C.shape[0] and 0 <= nv < C.shape[1]:
                                if Cabs[nu, nv] > 0.1:  # Only if coefficient is significant
                                    Cmod[nu, nv] *= (1.0 + comprehensive_alpha * 0.3 * S[i])

    Cmod *= sign
    out = _2d_idct(Cmod)
    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    return out

import numpy as np
import cv2
from scipy.fft import dct, idct  # Import Discrete Cosine Transform and its inverse

# ----------------------------
# Embedding hyperparameters
# ----------------------------
ALPHA = 0.017         # Base modulation strength (this will be scaled adaptively per coefficient)
MARK_SIZE = 1024       # Number of DCT coefficients to modify (length of the watermark)
SEED = 123             # Fixed seed for reproducibility (ensures we select the same coordinates every time)
MID_LO, MID_HI = 1, 100  # Candidate band: all (u,v) coordinates where 1 <= u+v <= 100 are candidates
REDUNDANCY_FACTOR = 2  # >1 spreads each bit to neighbor coefficients for extra robustness

def _load_watermark(input2, mark_size=MARK_SIZE):
    """
    Load the watermark as a binary vector of length 'mark_size'.
    - Accepts a .npy file path or any array-like object (e.g., list, np.array).
    - Tiles or truncates the input to ensure the output is exactly 'mark_size'.
    """
    # Load the raw watermark data
    if isinstance(input2, str):
        W = np.load(input2)  # Load from .npy file if input is a path
    else:
        W = np.array(input2) # Otherwise, convert the input (e.g., a list) to an array

    # Ensure the watermark is a 1D binary vector {0, 1}
    W = (W.flatten() % 2).astype(np.uint8)

    # Adjust the vector length to match MARK_SIZE
    if W.size < mark_size:
        # If too short, tile the vector to fill the required length
        W = np.tile(W, int(np.ceil(mark_size / W.size)))[:mark_size]
    else:
        # If too long, truncate it
        W = W[:mark_size]
    return W

# Orthonormal separable 2D-DCT/IDCT
def _2d_dct(x):  
    """Computes the 2D Discrete Cosine Transform (orthonormal version)."""
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def _2d_idct(X): 
    """Computes the 2D Inverse Discrete Cosine Transform (orthonormal version)."""
    return idct(idct(X, axis=1, norm='ortho'), axis=0, norm='ortho')

def _midband_coords(h, w, lo=MID_LO, hi=MID_HI):
    """
    Return all (u,v) coordinates whose taxicab frequency sum (u+v) lies in [lo, hi].
    This defines the candidate region where we can embed (low+mid+some high).
    """
    # Generate a list of all (u,v) pairs within the image dimensions
    # that satisfy the frequency sum constraint.
    return [(u, v) for u in range(h) for v in range(w) if lo <= (u + v) <= hi]

def _fixed_coords(h, w, k, seed=SEED, lo=MID_LO, hi=MID_HI):
    """
    Deterministically select 'k' coordinates with a frequency-aware sampling:
    - 35% from low (u+v ≤ 15)
    - 55% from mid (15 < u+v ≤ 50)
    - 10% from high (50 < u+v ≤ 100)
    - If not enough coords are available, top up from the global midband [lo, hi].
    The fixed RNG seed ensures reproducibility across runs.
    """
    # Use a local, seeded Random Number Generator for deterministic results
    rng_local = np.random.default_rng(seed)

    # Define the coordinate pools for each specific frequency band
    low_coords  = [(u, v) for u in range(h) for v in range(w) if 1 <= (u + v) <= 15]
    mid_coords  = [(u, v) for u in range(h) for v in range(w) if 15 < (u + v) <= 50]
    high_coords = [(u, v) for u in range(h) for v in range(w) if 50 < (u + v) <= 100]
    
    # Calculate the *target* number of samples from each band, capped by availability
    low_count  = min(int(round(k * 0.35)), len(low_coords))  # 35% low
    mid_count  = min(int(round(k * 0.55)), len(mid_coords))  # 55% mid
    high_count = min(k - low_count - mid_count, len(high_coords)) # 10% high (the remainder)

    selected_coords = []
    
    # Sample without replacement from the low band
    if low_count > 0:
        idx = rng_local.choice(len(low_coords), size=low_count, replace=False)
        selected_coords.extend([low_coords[i] for i in idx])
    
    # Sample without replacement from the mid band
    if mid_count > 0:
        idx = rng_local.choice(len(mid_coords), size=mid_count, replace=False)
        selected_coords.extend([mid_coords[i] for i in idx])
        
    # Sample without replacement from the high band
    if high_count > 0:
        idx = rng_local.choice(len(high_coords), size=high_count, replace=False)
        selected_coords.extend([high_coords[i] for i in idx])

    # If we still need more coordinates (e.g., bands were small),
    # top up from the general pool, avoiding duplicates.
    remaining = k - len(selected_coords)
    if remaining > 0:
        all_coords = _midband_coords(h, w, lo, hi)
        # Find coordinates that are in the general pool but not yet selected
        available = [c for c in all_coords if c not in selected_coords]
        
        if len(available) >= remaining:
            # Sample the remaining needed coordinates
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
    # 1. Load the original image
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")
    # Convert to float32 for mathematical operations
    imgf = img.astype(np.float32)

    # 2. Load watermark bits (W) and map them to bipolar symbols (S)
    W = _load_watermark(input2, MARK_SIZE)  # W is {0, 1}
    S = (2.0 * W.astype(np.float32) - 1.0)  # S is {-1, +1}

    # 3. Compute 2D-DCT and separate magnitude from sign
    C = _2d_dct(imgf)
    sign = np.sign(C)  # Store the original signs (phase)
    Cabs = np.abs(C)   # We will modify the magnitudes

    # 4. Get the deterministic set of coordinates for embedding
    locs = _fixed_coords(*C.shape, k=MARK_SIZE, seed=SEED)

    # 5. Apply the watermark modulation
    # Work on a copy of the magnitudes
    Cmod = Cabs.copy()
    
    # Iterate over each selected coordinate (u,v) and its corresponding symbol S[i]
    for i, (u, v) in enumerate(locs):
        coeff_magnitude = Cabs[u, v]
        freq_sum = u + v # Use (u+v) as a simple proxy for frequency

        # Only modify coefficients that have some energy.
        # This avoids numerical issues and visual artifacts in flat areas.
        if coeff_magnitude > 1.0:
            
            # --- 5a. Calculate adaptive strength (alpha) ---
            
            # Frequency-dependent factor: adjust strength based on frequency band
            if freq_sum <= 15:
                freq_factor = 1.2  # Boost strength in stable low frequencies
            elif freq_sum <= 50:
                freq_factor = 1.0  # Baseline strength in mid frequencies
            else:
                freq_factor = 1.1  # Slightly boost in high frequencies

            # Magnitude-dependent factor: embed more strongly in larger coefficients
            # (based on Weber's law: changes to large values are less perceptible)
            magnitude_factor = 1.0 + 0.1 * np.log(1.0 + coeff_magnitude)

            # Position-dependent factor: very mild tapering across the matrix
            u_norm = u / C.shape[0]
            v_norm = v / C.shape[1] # Normalized position (0 to 1)
            position_factor = 1.0 - 0.01 * (u_norm + v_norm)

            # Combine all factors to get the final strength for this coefficient
            comprehensive_alpha = ALPHA * freq_factor * magnitude_factor * position_factor

            # --- 5b. Apply multiplicative modulation ---
            # C_wm = C_orig * (1 + alpha * S)
            
            for redundancy in range(REDUNDANCY_FACTOR):
                if redundancy == 0:
                    # Primary write: modify the target coefficient
                    Cmod[u, v] *= (1.0 + comprehensive_alpha * S[i])
                else:
                    # Redundant write: spread the *same* symbol to neighbors
                    # This adds robustness against desynchronization (e.g., cropping, scaling)
                    offset = redundancy
                    for du in [-offset, 0, offset]: # Loop over 3x3 (or 5x5...) neighborhood
                        for dv in [-offset, 0, offset]:
                            if du == 0 and dv == 0:
                                continue # Skip the center coefficient (already done)
                            
                            nu, nv = u + du, v + dv # Neighbor coordinates

                            # Check if neighbor is within image boundaries
                            if 0 <= nu < C.shape[0] and 0 <= nv < C.shape[1]:
                                # Only modify non-trivial neighbors
                                if Cabs[nu, nv] > 0.1: 
                                    # Apply modulation with reduced strength (0.3)
                                    Cmod[nu, nv] *= (1.0 + comprehensive_alpha * 0.3 * S[i])

    # 6. Reconstruct the watermarked image
    # Re-apply the original signs to the modified magnitudes
    Cmod *= sign
    # Perform Inverse 2D-DCT to return to the spatial domain
    out = _2d_idct(Cmod)

    # 7. Finalize: Clip, round, and convert to 8-bit integer
    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    return out
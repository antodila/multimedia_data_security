import numpy as np
from scipy.signal import convolve2d

# --- CSF HELPER FUNCTIONS (THESE WERE MISSING) ---

def _csffun(u, v):
    """Contrast Sensitivity Function (CSF) model."""
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
    """Generate the CSF matrix in the frequency domain."""
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    Fmat = _csffun(u, v)
    return Fmat

def _get_csf_filter():
    """Compute the spatial domain filter by taking the IFFT of the CSF matrix."""
    Fmat = _csfmat()
    # Inverse FFT to get spatial filter, shifting quadrants for correct alignment
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)

# --- MAIN WPSNR FUNCTION ---

def wpsnr(image_a, image_b):
    """
    CSF-weighted PSNR. Returns a very large value if images are identical.
    """
    if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
        raise TypeError("Input images must be NumPy arrays.")
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Normalize images to [0, 1] float if they are uint8
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

    # Handle identical images
    if np.array_equal(A, B):
        return 9999999.0

    # Calculate error and apply CSF weighting
    error_image = A - B
    csf_filter = _get_csf_filter()
    # Convolve error with CSF spatial filter
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    
    # Calculate Weighted Mean Squared Error (WMSE)
    wmse = np.mean(weighted_error**2)
    if wmse == 0:
        return 9999999.0
        
    # Convert to dB scale
    decibels = 20 * np.log10(1.0 / np.sqrt(wmse))
    return decibels
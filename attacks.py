import numpy as np
import cv2

# ==========================
# Basic (single) attack functions
# ==========================

def attack_jpeg(img, qf=20):
    """
    JPEG compression attack.
    Args:
        img: Grayscale input image (numpy array).
        qf: JPEG quality factor (lower = stronger compression).
    Returns:
        Decompressed grayscale image after JPEG encoding/decoding.
    """
    # Define the JPEG quality parameter for the encoder
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)]
    
    # Encode the image into a JPEG format in a memory buffer
    ok, buf = cv2.imencode(".jpg", img, enc)
    
    if not ok:
        return img  # Fallback: return original if encoding fails
        
    # Decode the image from the memory buffer back into a numpy array
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def attack_awgn(img, sigma=25):
    """
    Additive White Gaussian Noise (AWGN) attack.
    Args:
        img: Grayscale input image.
        sigma: Standard deviation of the Gaussian noise.
    Returns:
        Noisy image clipped to [0,255].
    """
    # Generate a noise matrix with the same shape as the image
    # The noise is drawn from a normal distribution (mean=0, std=sigma)
    g = np.random.normal(0, sigma, img.shape).astype(np.float32)
    
    # Add the noise to the image (converted to float for the operation)
    out = img.astype(np.float32) + g
    
    # Clip the values to the valid [0, 255] range and convert back to uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def attack_blur(img, ksize=13):
    """
    Gaussian blur attack.
    Args:
        ksize: Kernel size (odd number). Higher values → stronger blur.
    Returns:
        Blurred image.
    """
    # Ensure the kernel size is odd, as required by OpenCV
    k = ksize if ksize % 2 else ksize + 1
    
    # Apply Gaussian blur with a (k, k) kernel. Sigma is 0, so it's auto-calculated.
    return cv2.GaussianBlur(img, (k, k), 0)


def attack_median(img, ksize=11):
    """
    Median filter attack.
    Useful to remove salt-and-pepper noise but can distort watermark structure.
    """
    # Ensure the kernel size is odd, as required by OpenCV
    k = ksize if ksize % 2 else ksize + 1
    
    # Apply the median filter, which replaces each pixel with the median of its neighbors
    return cv2.medianBlur(img, k)


def attack_resize(img, scale=0.25):
    """
    Resize attack (downscale + upscale).
    Reduces spatial resolution and removes high-frequency watermark components.
    Args:
        scale: Downscaling ratio (0 < scale < 1).
    Returns:
        Image resized back to original size after downscaling.
    """
    # Get original image dimensions
    h, w = img.shape
    
    # Calculate new, smaller dimensions, ensuring they are at least 1 pixel
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    
    # Downscale the image using linear interpolation
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Upscale the small image back to the original dimensions
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_sharp(img, amount=1.0, radius=1):
    """
    Sharpening filter using unsharp masking.
    Args:
        amount: Sharpening strength.
        radius: Gaussian blur radius.
    Returns:
        Sharpened image with enhanced edges.
    """
    # Create the "unsharp" mask by blurring the original image
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    
    # Apply unsharp masking: Img = Orig + amount * (Orig - Blur)
    # This is implemented using cv2.addWeighted: (1+amount)*Orig - amount*Blur
    out = cv2.addWeighted(img.astype(np.float32), 1.0 + amount,
                          blur.astype(np.float32), -amount, 0.0)
                          
    # Clip to valid range and convert back to uint8
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ==========================
# Composite / Multi-step attack strategies
# These functions chain multiple single attacks to create a more robust test.
# ==========================

def attack_strategy_1_stealth(img):
    """
    Stealth attack:
    Mild blur + 20% downscale + low Gaussian noise + median cleanup.
    Balances imperceptibility and watermark degradation.
    """
    img = cv2.GaussianBlur(img, (3, 3), 0) # 1. Apply a very mild blur
    h, w = img.shape
    # 2. Downscale by 20% (to 80% of original size)
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    # 3. Upscale back to original size
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    # 4. Add light noise
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # 5. Use a median filter to clean up some of the noise (and further attack)
    img = cv2.medianBlur(img, 3)
    return img


def attack_strategy_2_brutal(img):
    """
    Brutal attack:
    Heavy blur + strong downscale (70%) + high Gaussian noise.
    Aims to fully destroy watermark structure.
    """
    img = cv2.GaussianBlur(img, (15, 15), 0) # 1. Apply a very heavy blur
    h, w = img.shape
    # 2. Downscale by 70% (to 30% of original size)
    small = cv2.resize(img, (int(w*0.3), int(h*0.3)), cv2.INTER_LINEAR)
    # 3. Upscale back
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    # 4. Add strong noise
    noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def attack_strategy_3_smart(img):
    """
    Smart attack:
    Moderate blur + JPEG recompression + median filter.
    Designed to mimic realistic image editing pipelines.
    """
    img = cv2.GaussianBlur(img, (7, 7), 0) # 1. Apply moderate blur
    
    # 2. Apply strong JPEG compression (Quality Factor 30)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    ok, buf = cv2.imencode(".jpg", img, enc)
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        
    # 3. Apply a median filter
    img = cv2.medianBlur(img, 5)
    return img


def attack_strategy_4_chaos(img):
    """
    Chaos attack:
    Randomly picks one of several multi-operation attack combinations.
    Each combination mixes different types of degradations.
    """
    # Define a list of attack "chains" (each chain is a list of functions)
    attack_combinations = [
        # Chain 1: Blur -> Noise -> JPEG
        [lambda x: cv2.GaussianBlur(x, (3, 3), 0),
         lambda x: attack_awgn(x, 8),
         lambda x: attack_jpeg(x, 60)],
        # Chain 2: Median Filter -> Resize -> Noise
        [lambda x: cv2.medianBlur(x, 5),
         lambda x: attack_resize(x, 0.7),
         lambda x: attack_awgn(x, 12)],
        # Chain 3: Blur -> JPEG -> Median Filter
        [lambda x: cv2.GaussianBlur(x, (7, 7), 0),
         lambda x: attack_jpeg(x, 40),
         lambda x: cv2.medianBlur(x, 3)],
        # Chain 4: Blur -> Noise -> Resize
        [lambda x: cv2.GaussianBlur(x, (9, 9), 0),
         lambda x: attack_awgn(x, 6),
         lambda x: attack_resize(x, 0.5)]
    ]
    # Randomly pick one of the chains from the list
    chosen_combination = np.random.choice(len(attack_combinations))
    attacks = attack_combinations[chosen_combination]
    
    # Apply each function (attack) in the chosen chain, in order
    for attack in attacks:
        img = attack(img)
    return img


def attack_strategy_5_precision(img):
    """
    Precision attack:
    Smooth blur + 20% downscale + medium noise.
    Keeps edges smooth while degrading watermark mid-bands.
    """
    img = cv2.GaussianBlur(img, (11, 11), 0) # 1. Strong blur
    h, w = img.shape
    # 2. Mild downscale (to 80%)
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    # 3. Upscale back
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    # 4. Add medium noise
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def attack_strategy_6_wave(img):
    """
    Wave attack:
    Cascade of multiple mild distortions (blur + noise + compression + resize).
    Designed to simulate subtle but cumulative degradations.
    """
    img = cv2.GaussianBlur(img, (3, 3), 0) # 1. Mild blur
    img = attack_awgn(img, 5)             # 2. Light noise
    img = cv2.medianBlur(img, 5)          # 3. Median filter
    img = attack_jpeg(img, 50)            # 4. Medium JPEG compression
    img = cv2.GaussianBlur(img, (9, 9), 0) # 5. Stronger blur
    img = attack_resize(img, 0.5)         # 6. Strong resize
    return img


def attack_strategy_7_counter_intuitive(img):
    """
    Counter-intuitive attack:
    JPEG compression + heavy blur + light downscale.
    Keeps perceptual smoothness high but reduces watermark contrast.
    """
    img = attack_jpeg(img, 80) # 1. High-quality JPEG compression
    img = cv2.GaussianBlur(img, (13, 13), 0) # 2. Heavy blur
    h, w = img.shape
    # 3. Very light downscale (to 90%)
    small = cv2.resize(img, (int(w*0.9), int(h*0.9)), cv2.INTER_LINEAR)
    # 4. Upscale back
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    return img


def attack_strategy_8_frequency_hunter(img):
    """
    Frequency-hunter attack:
    Progressive multi-scale blur + low Gaussian noise + median filter.
    Focused on reducing mid- and high-frequency DCT coefficients.
    """
    # 1. Apply blur with progressively larger kernels
    for ksize in [3, 5, 7]:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    # 2. Add low noise
    noise = np.random.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. Clean up with a mild median filter
    img = cv2.medianBlur(img, 3)
    return img


def attack_strategy_9_surgical(img):
    """
    Surgical attack:
    Mild blur + light JPEG + minimal noise.
    Attempts to remove watermark gently while maintaining high quality.
    """
    img = cv2.GaussianBlur(img, (3, 3), 0) # 1. Mild blur
    img = attack_jpeg(img, 50)            # 2. Medium JPEG compression
    # 3. Add minimal noise
    noise = np.random.normal(0, 4, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# ==========================
# Unified dispatcher
# ==========================

def attacks(input1, attack_name, param_array=None):
    """
    Generic attack dispatcher.
    Loads image and applies selected attack by name.

    Args:
        input1: Path to input grayscale image.
        attack_name: String identifier (e.g., 'jpeg', 'awgn', 'strategy_3_smart').
        param_array: Optional list of parameters for the attack.
    Returns:
        Modified image (numpy array, uint8).
    """
    # Load the source image in grayscale
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")

    # --- Strategy-based attacks (no parameters needed) ---
    # This block routes the attack_name string to the correct strategy function.
    if attack_name == "strategy_1_stealth":
        return attack_strategy_1_stealth(img)
    elif attack_name == "strategy_2_brutal":
        return attack_strategy_2_brutal(img)
    elif attack_name == "strategy_3_smart":
        return attack_strategy_3_smart(img)
    elif attack_name == "strategy_4_chaos":
        return attack_strategy_4_chaos(img)
    elif attack_name == "strategy_5_precision":
        return attack_strategy_5_precision(img)
    elif attack_name == "strategy_6_wave":
        return attack_strategy_6_wave(img)
    elif attack_name == "strategy_7_counter_intuitive":
        return attack_strategy_7_counter_intuitive(img)
    elif attack_name == "strategy_8_frequency_hunter":
        return attack_strategy_8_frequency_hunter(img)
    elif attack_name == "strategy_9_surgical":
        return attack_strategy_9_surgical(img)

    # --- Parameterized single attacks ---
    # This block handles single attacks that require parameters from param_array.
    elif attack_name == "sharp":
        # Safely extract parameters from the list, providing defaults if not present
        amount = param_array[0] if param_array and len(param_array) > 0 else 1.0
        radius = param_array[1] if param_array and len(param_array) > 1 else 1
        return attack_sharp(img, amount=amount, radius=radius)

    elif attack_name == "jpeg":
        # Use the first parameter as 'qf', or a default value
        qf = param_array[0] if param_array else 20
        return attack_jpeg(img, qf)
    elif attack_name == "awgn":
        # Use the first parameter as 'sigma', or a default value
        sigma = param_array[0] if param_array else 25
        return attack_awgn(img, sigma)
    elif attack_name == "blur":
        # Use the first parameter as 'ksize', or a default value
        ksize = param_array[0] if param_array else 13
        return attack_blur(img, ksize)
    elif attack_name == "median":
        # Use the first parameter as 'ksize', or a default value
        ksize = param_array[0] if param_array else 11
        return attack_median(img, ksize)
    elif attack_name == "resize":
        # Use the first parameter as 'scale', or a default value
        scale = param_array[0] if param_array else 0.25
        return attack_resize(img, scale)

    # Default fallback
    else:
        # If the attack_name is not recognized, apply the 'stealth' strategy as a default
        return attack_strategy_1_stealth(img)
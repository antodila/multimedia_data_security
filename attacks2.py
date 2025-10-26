import numpy as np
from PIL import Image
import os
from scipy import ndimage

def attacks(input1, attack_name, param_array):
    """
    Apply various attacks to watermarked images
    
    Args:
        input1 (str): Path to watermarked image
        attack_name (str): Type of attack
        param_array (list): Parameters for the attack
    
    Returns:
        numpy.ndarray: Attacked image as uint8 array
    """
    # Load the watermarked image
    image = np.array(Image.open(input1).convert('L'))
    
    if attack_name.lower() == "jpeg":
        # JPEG compression attack
        quality = int(param_array[0]) if param_array else 90
        quality = max(1, min(100, quality))
        
        img_pil = Image.fromarray(image)
        temp_path = "temp_jpeg_attack.jpg"
        img_pil.save(temp_path, "JPEG", quality=quality)
        attacked_image = np.array(Image.open(temp_path))
        os.remove(temp_path)
        
    elif attack_name.lower() == "awgn":
        # Additive White Gaussian Noise attack
        noise_std = param_array[0] if param_array else 25
        noise = np.random.normal(0, noise_std, image.shape)
        attacked_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    elif attack_name.lower() == "blur":
        # Gaussian blur attack
        kernel_size = int(param_array[0]) if param_array else 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = kernel_size / 3.0
        attacked_image = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
        attacked_image = np.clip(attacked_image, 0, 255).astype(np.uint8)
        
    elif attack_name.lower() == "median":
        # Median filtering attack
        kernel_size = int(param_array[0]) if param_array else 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        attacked_image = ndimage.median_filter(image, size=kernel_size)
        
    elif attack_name.lower() == "sharp":
        # Sharpening attack
        sigma = param_array[0] if param_array else 1.0
        amount = param_array[1] if len(param_array) > 1 else 0.5
        
        blurred = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
        mask = image.astype(np.float32) - blurred
        attacked_image = image.astype(np.float32) + amount * mask
        attacked_image = np.clip(attacked_image, 0, 255).astype(np.uint8)
        
    elif attack_name.lower() == "resize":
        # Resizing attack
        scale_factor = param_array[0] if param_array else 0.5
        scale_factor = max(0.1, min(2.0, scale_factor))
        
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        img_pil = Image.fromarray(image)
        resized_down = img_pil.resize((new_w, new_h), Image.LANCZOS)
        attacked_image = np.array(resized_down.resize((w, h), Image.LANCZOS))
        
    else:
        print(f"Warning: Unknown attack '{attack_name}'. Returning original image.")
        attacked_image = image.copy()
    
    return attacked_image

import numpy as np
from PIL import Image

def detection(input1, input2, input3):
    """
    Non-blind watermark detection function
    
    Args:
        input1 (str): Path to original image
        input2 (str): Path to watermarked image
        input3 (str): Path to attacked image
    
    Returns:
        tuple: (detection_result, WPSNR)
    """
    # Read all images
    original = np.array(Image.open(input1).convert('L')).astype(np.float32)
    watermarked = np.array(Image.open(input2).convert('L')).astype(np.float32)
    attacked = np.array(Image.open(input3).convert('L')).astype(np.float32)
    
    # Extract watermarks
    watermark_ref = extract_watermark_svd(original, watermarked)
    watermark_attacked = extract_watermark_svd(original, attacked)
    
    # Compute similarity and WPSNR
    similarity = compute_similarity(watermark_ref, watermark_attacked)
    wpsnr = compute_wpsnr(watermarked, attacked)
    
    # Detection threshold (from ROC analysis)
    detection_threshold = 0.7
    detection_result = 1 if similarity >= detection_threshold else 0
    
    return detection_result, wpsnr


def extract_watermark_svd(original, processed):
    """Extract watermark using SVD comparison"""
    rows, cols = original.shape
    block_size = 8
    watermark = []
    
    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            orig_block = original[i:i+block_size, j:j+block_size]
            proc_block = processed[i:i+block_size, j:j+block_size]
            
            _, s_orig, _ = np.linalg.svd(orig_block)
            _, s_proc, _ = np.linalg.svd(proc_block)
            
            # Compare largest singular values
            if s_proc[0] >= s_orig[0]:
                watermark.append(1)
            else:
                watermark.append(0)
            
            if len(watermark) >= 1024:
                break
                
        if len(watermark) >= 1024:
            break
    
    if len(watermark) < 1024:
        watermark.extend([0] * (1024 - len(watermark)))
    
    return np.array(watermark[:1024])


def compute_similarity(w1, w2):
    """Compute normalized correlation"""
    min_len = min(len(w1), len(w2))
    w1 = np.array(w1[:min_len])
    w2 = np.array(w2[:min_len])
    
    if np.std(w1) == 0 or np.std(w2) == 0:
        return 1.0 if np.array_equal(w1, w2) else 0.0
    
    correlation_matrix = np.corrcoef(w1, w2)
    correlation = correlation_matrix[0, 1]
    
    if np.isnan(correlation):
        correlation = 0.0
    
    return abs(correlation)


def compute_wpsnr(img1, img2):
    """Compute Weighted PSNR"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100.0
    
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

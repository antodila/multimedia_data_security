import numpy as np
from PIL import Image
import os
from scipy import ndimage

def embedding(input1, input2):
    """
    SVD-based watermark embedding function
    
    Args:
        input1 (str): Path to the original image
        input2 (str): Path to the watermark (.npy file)
    
    Returns:
        numpy.ndarray: Watermarked image as uint8 array
    """
    # Read the host image
    host_image = np.array(Image.open(input1).convert('L')).astype(np.float32)
    
    # Load watermark (binary array)
    watermark = np.load(input2).astype(np.float32)
    
    # Ensure watermark is 1024 bits
    if len(watermark) > 1024:
        watermark = watermark[:1024]
    elif len(watermark) < 1024:
        watermark = np.pad(watermark, (0, 1024 - len(watermark)), 'constant')
    
    rows, cols = host_image.shape
    block_size = 8
    alpha = 0.03  # Embedding strength
    
    watermarked_image = host_image.copy()
    watermark_idx = 0
    
    # Block-based SVD embedding
    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            if watermark_idx >= len(watermark):
                break
                
            block = host_image[i:i+block_size, j:j+block_size]
            U, s, Vt = np.linalg.svd(block)
            
            # Embed in largest singular value
            if watermark[watermark_idx] == 1:
                s[0] = s[0] * (1 + alpha)
            else:
                s[0] = s[0] * (1 - alpha)
            
            watermarked_block = np.dot(U, np.dot(np.diag(s), Vt))
            watermarked_image[i:i+block_size, j:j+block_size] = watermarked_block
            
            watermark_idx += 1
            
        if watermark_idx >= len(watermark):
            break
    
    return watermarked_image.astype(np.uint8)
import numpy as np
from PIL import Image
import os
from scipy import ndimage

def embedding(input1, input2):
    """
    SVD-based watermark embedding function
    
    Args:
        input1 (str): Path to the original image
        input2 (str): Path to the watermark (.npy file)
    
    Returns:
        numpy.ndarray: Watermarked image as uint8 array
    """
    # Read the host image
    host_image = np.array(Image.open(input1).convert('L')).astype(np.float32)
    
    # Load watermark (binary array)
    watermark = np.load(input2).astype(np.float32)
    
    # Ensure watermark is 1024 bits
    if len(watermark) > 1024:
        watermark = watermark[:1024]
    elif len(watermark) < 1024:
        watermark = np.pad(watermark, (0, 1024 - len(watermark)), 'constant')
    
    rows, cols = host_image.shape
    block_size = 8
    alpha = 0.03  # Embedding strength
    
    watermarked_image = host_image.copy()
    watermark_idx = 0
    
    # Block-based SVD embedding
    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            if watermark_idx >= len(watermark):
                break
                
            block = host_image[i:i+block_size, j:j+block_size]
            U, s, Vt = np.linalg.svd(block)
            
            # Embed in largest singular value
            if watermark[watermark_idx] == 1:
                s[0] = s[0] * (1 + alpha)
            else:
                s[0] = s[0] * (1 - alpha)
            
            watermarked_block = np.dot(U, np.dot(np.diag(s), Vt))
            watermarked_image[i:i+block_size, j:j+block_size] = watermarked_block
            
            watermark_idx += 1
            
        if watermark_idx >= len(watermark):
            break
    
    return watermarked_image.astype(np.uint8)

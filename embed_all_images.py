#!/usr/bin/env python3
"""
Embed watermarks in all 101 images following README.md workflow
"""

import os
import cv2
import numpy as np
from pathlib import Path
from embedding import embedding

def embed_all_images():
    print("=== Embedding Watermarks in All 101 Images ===")
    
    # Setup paths
    images_dir = Path("images")
    output_dir = Path("watermarked_images")
    output_dir.mkdir(exist_ok=True)
    
    mark_file = "mark.npy"
    
    # Get all BMP images
    image_files = sorted([f for f in images_dir.glob("*.bmp")])
    print(f"Found {len(image_files)} images to process")
    
    results = []
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/101: {img_path.name}")
            
            # Embed watermark
            wm_img = embedding(str(img_path), mark_file)
            
            # Save watermarked image with proper naming convention
            wm_path = output_dir / f"shadowmark_{img_path.stem}.bmp"
            cv2.imwrite(str(wm_path), wm_img)
            
            results.append({
                'original': str(img_path),
                'watermarked': str(wm_path),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                'original': str(img_path),
                'watermarked': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    import json
    with open('embedding_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n=== Embedding Complete ===")
    print(f"Successfully watermarked: {successful}/101 images")
    print(f"Results saved to: embedding_results.json")
    print(f"Watermarked images saved to: {output_dir}/")
    
    return results

if __name__ == "__main__":
    results = embed_all_images()

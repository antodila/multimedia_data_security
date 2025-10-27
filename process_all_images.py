#!/usr/bin/env python3
"""
Comprehensive workflow for processing all 101 sample images
Based on README.md instructions
"""

import os
import cv2
import numpy as np
import json
import time
from pathlib import Path
from embedding import embedding
from detection_shadowmark import detection, _extract_ratio_vector, _similarity, wpsnr
import attacks as A

def main():
    print("=== Shadowmark Comprehensive Workflow ===")
    print("Processing all 101 sample images...")
    
    # Setup paths
    sample_dir = Path("images/sample-images")
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    
    mark_file = "mark.npy"
    
    # Verify watermark exists
    if not os.path.exists(mark_file):
        print("Generating watermark...")
        np.save(mark_file, (np.random.rand(1024) > 0.5).astype('uint8'))
        print("Dummy mark.npy created")
    
    # Get all sample images
    sample_images = sorted([f for f in sample_dir.glob("*.bmp")])
    print(f"Found {len(sample_images)} sample images")
    
    results = {
        'embedding_results': [],
        'smoke_test_results': [],
        'detection_results': [],
        'attack_results': []
    }
    
    # Step 1: Embed watermark in all images
    print("\n=== Step 1: Embedding Watermarks ===")
    watermarked_images = []
    
    for i, img_path in enumerate(sample_images):
        try:
            print(f"Processing {i+1}/101: {img_path.name}")
            
            # Embed watermark
            wm_img = embedding(str(img_path), mark_file)
            
            # Save watermarked image
            wm_path = output_dir / f"shadowmark_{img_path.stem}.bmp"
            cv2.imwrite(str(wm_path), wm_img)
            
            watermarked_images.append((img_path, wm_path))
            
            results['embedding_results'].append({
                'original': str(img_path),
                'watermarked': str(wm_path),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results['embedding_results'].append({
                'original': str(img_path),
                'watermarked': None,
                'status': 'error',
                'error': str(e)
            })
    
    print(f"Successfully watermarked {len(watermarked_images)} images")
    
    # Step 2: Smoke tests on subset of images
    print("\n=== Step 2: Smoke Tests ===")
    test_images = watermarked_images[:10]  # Test first 10 images
    
    for orig_path, wm_path in test_images:
        try:
            print(f"Smoke test: {orig_path.name}")
            
            # Load images
            orig = cv2.imread(str(orig_path), 0)
            wm = cv2.imread(str(wm_path), 0)
            
            # True Positive: self similarity
            v_wm = _extract_ratio_vector(wm, orig)
            tp_sim = _similarity(v_wm, v_wm)
            
            # True Negative: wm vs clean
            v_clean = _extract_ratio_vector(orig, orig)
            tn_sim = _similarity(v_wm, v_clean)
            
            # Test attacks
            attack_results = {}
            candidates = [
                ("jpeg", {"qf": 60}),
                ("awgn", {"sigma": 6}),
                ("blur", {"ksize": 5}),
                ("median", {"ksize": 3}),
                ("resize", {"scale": 0.6}),
            ]
            
            for name, params in candidates:
                if name == "jpeg":   att = A.attack_jpeg(wm, **params)
                elif name == "awgn":   att = A.attack_awgn(wm, **params)
                elif name == "blur":   att = A.attack_blur(wm, **params)
                elif name == "median": att = A.attack_median(wm, **params)
                elif name == "resize": att = A.attack_resize(wm, **params)
                
                sim = _similarity(v_wm, _extract_ratio_vector(att, orig))
                wps = wpsnr(wm, att)
                attack_results[name] = {'sim': sim, 'wpsnr': wps}
            
            results['smoke_test_results'].append({
                'image': orig_path.name,
                'tp_similarity': tp_sim,
                'tn_similarity': tn_sim,
                'attacks': attack_results
            })
            
            print(f"  TP sim: {tp_sim:.3f}, TN sim: {tn_sim:.3f}")
            
        except Exception as e:
            print(f"Smoke test error for {orig_path.name}: {e}")
            results['smoke_test_results'].append({
                'image': orig_path.name,
                'error': str(e)
            })
    
    # Step 3: Detection tests
    print("\n=== Step 3: Detection Tests ===")
    test_images = watermarked_images[:5]  # Test first 5 images
    
    for orig_path, wm_path in test_images:
        try:
            print(f"Detection test: {orig_path.name}")
            
            # Test 1: Watermarked image (should detect)
            presence, wpsnr_val = detection(str(orig_path), str(wm_path), str(wm_path))
            print(f"  Watermarked -> presence: {presence}, WPSNR: {wpsnr_val}")
            
            # Test 2: Clean image (should NOT detect)
            presence_clean, wpsnr_clean = detection(str(orig_path), str(wm_path), str(orig_path))
            print(f"  Clean -> presence: {presence_clean}, WPSNR: {wpsnr_clean}")
            
            # Test 3: Destroyed image
            wm = cv2.imread(str(wm_path), 0)
            destroyed = A.attack_awgn(wm, sigma=30)
            destroyed_path = output_dir / f"destroyed_{orig_path.stem}.bmp"
            cv2.imwrite(str(destroyed_path), destroyed)
            
            presence_destroyed, wpsnr_destroyed = detection(str(orig_path), str(wm_path), str(destroyed_path))
            print(f"  Destroyed -> presence: {presence_destroyed}, WPSNR: {wpsnr_destroyed}")
            
            results['detection_results'].append({
                'image': orig_path.name,
                'watermarked': {'presence': presence, 'wpsnr': wpsnr_val},
                'clean': {'presence': presence_clean, 'wpsnr': wpsnr_clean},
                'destroyed': {'presence': presence_destroyed, 'wpsnr': wpsnr_destroyed}
            })
            
        except Exception as e:
            print(f"Detection test error for {orig_path.name}: {e}")
            results['detection_results'].append({
                'image': orig_path.name,
                'error': str(e)
            })
    
    # Step 4: Attack validation
    print("\n=== Step 4: Attack Validation ===")
    test_images = watermarked_images[:10]  # Test first 10 images
    
    for orig_path, wm_path in test_images:
        try:
            print(f"Attack validation: {orig_path.name}")
            
            wm = cv2.imread(str(wm_path), 0)
            orig = cv2.imread(str(orig_path), 0)
            
            # Test resize + JPEG combination (known effective attack)
            attacked = A.attack_resize(wm, scale=0.6)
            attacked = A.attack_jpeg(attacked, qf=70)
            
            attacked_path = output_dir / f"attacked_{orig_path.stem}_resize06_jpeg70.bmp"
            cv2.imwrite(str(attacked_path), attacked)
            
            presence, wpsnr_val = detection(str(orig_path), str(wm_path), str(attacked_path))
            
            results['attack_results'].append({
                'image': orig_path.name,
                'attack': 'resize_0.6_jpeg_70',
                'presence': presence,
                'wpsnr': wpsnr_val,
                'valid_attack': presence == 0 and wpsnr_val >= 35
            })
            
            print(f"  Resize+JPEG -> presence: {presence}, WPSNR: {wpsnr_val}, Valid: {presence == 0 and wpsnr_val >= 35}")
            
        except Exception as e:
            print(f"Attack validation error for {orig_path.name}: {e}")
            results['attack_results'].append({
                'image': orig_path.name,
                'error': str(e)
            })
    
    # Save results
    with open('comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(watermarked_images)} images")
    print(f"Results saved to comprehensive_results.json")
    print(f"Watermarked images saved to {output_dir}/")
    
    return results

if __name__ == "__main__":
    results = main()

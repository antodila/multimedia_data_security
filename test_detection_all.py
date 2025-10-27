#!/usr/bin/env python3
"""
Test detection on all 101 images following README.md workflow
"""

import os
import cv2
import json
import time
from pathlib import Path
from detection_shadowmark import detection

def test_detection_all_images():
    print("=== Testing Detection on All 101 Images ===")
    
    # Setup paths
    images_dir = Path("images")
    watermarked_dir = Path("watermarked_images")
    
    # Get all image pairs
    image_files = sorted([f for f in images_dir.glob("*.bmp")])
    print(f"Found {len(image_files)} images to test")
    
    results = []
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"Testing {i+1}/101: {img_path.name}")
            
            # Find corresponding watermarked image
            wm_path = watermarked_dir / f"shadowmark_{img_path.stem}.bmp"
            
            if not wm_path.exists():
                print(f"Warning: Watermarked image not found for {img_path.name}")
                continue
            
            # Test 1: Watermarked image (should detect)
            t0 = time.time()
            presence_wm, wpsnr_wm = detection(str(img_path), str(wm_path), str(wm_path))
            detection_time = time.time() - t0
            total_time += detection_time
            
            # Test 2: Clean image (should NOT detect)
            presence_clean, wpsnr_clean = detection(str(img_path), str(wm_path), str(img_path))
            
            results.append({
                'image': img_path.name,
                'watermarked_detection': {
                    'presence': presence_wm,
                    'wpsnr': wpsnr_wm,
                    'time': detection_time
                },
                'clean_detection': {
                    'presence': presence_clean,
                    'wpsnr': wpsnr_clean
                },
                'status': 'success'
            })
            
            print(f"  Watermarked: presence={presence_wm}, WPSNR={wpsnr_wm:.2f}, time={detection_time:.3f}s")
            print(f"  Clean: presence={presence_clean}, WPSNR={wpsnr_clean:.2f}")
            
        except Exception as e:
            print(f"Error testing {img_path.name}: {e}")
            results.append({
                'image': img_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    with open('detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate statistics
    successful_tests = [r for r in results if r['status'] == 'success']
    avg_time = total_time / len(successful_tests) if successful_tests else 0
    
    correct_wm_detections = sum(1 for r in successful_tests if r['watermarked_detection']['presence'] == 1)
    correct_clean_detections = sum(1 for r in successful_tests if r['clean_detection']['presence'] == 0)
    
    print(f"\n=== Detection Test Complete ===")
    print(f"Successful tests: {len(successful_tests)}/101")
    print(f"Average detection time: {avg_time:.3f} seconds")
    print(f"Correct watermarked detections: {correct_wm_detections}/{len(successful_tests)}")
    print(f"Correct clean detections: {correct_clean_detections}/{len(successful_tests)}")
    print(f"Results saved to: detection_results.json")
    
    return results

if __name__ == "__main__":
    results = test_detection_all_images()

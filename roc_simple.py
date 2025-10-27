#!/usr/bin/env python3
"""
Simplified ROC computation for all 101 sample images
Avoids sklearn dependency issues
"""

import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from embedding import embedding
from detection_shadowmark import _extract_ratio_vector, _similarity
import attacks as A

def compute_roc_curve(labels, scores):
    """Simple ROC curve computation without sklearn"""
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Remove duplicate thresholds
    unique_thresholds = np.unique(sorted_scores)
    unique_thresholds = np.append(unique_thresholds, unique_thresholds[-1] - 1e-6)
    
    fpr = []
    tpr = []
    
    for threshold in unique_thresholds:
        # Predictions: 1 if score >= threshold, 0 otherwise
        predictions = (sorted_scores >= threshold).astype(int)
        
        # Calculate TP, FP, TN, FN
        tp = np.sum((predictions == 1) & (sorted_labels == 1))
        fp = np.sum((predictions == 1) & (sorted_labels == 0))
        tn = np.sum((predictions == 0) & (sorted_labels == 0))
        fn = np.sum((predictions == 0) & (sorted_labels == 1))
        
        # Calculate rates
        if fp + tn > 0:
            fpr_val = fp / (fp + tn)
        else:
            fpr_val = 0
            
        if tp + fn > 0:
            tpr_val = tp / (tp + fn)
        else:
            tpr_val = 0
            
        fpr.append(fpr_val)
        tpr.append(tpr_val)
    
    return np.array(fpr), np.array(tpr), unique_thresholds

def compute_auc(fpr, tpr):
    """Compute AUC using trapezoidal rule"""
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_sorted)):
        auc += (fpr_sorted[i] - fpr_sorted[i-1]) * (tpr_sorted[i] + tpr_sorted[i-1]) / 2
    
    return auc

def main():
    print("=== ROC Computation for All 101 Sample Images ===")
    
    # Setup paths
    sample_dir = Path("images/sample-images")
    mark_file = "mark.npy"
    
    # Get all sample images
    sample_images = sorted([f for f in sample_dir.glob("*.bmp")])
    print(f"Found {len(sample_images)} sample images")
    
    scores = []
    labels = []
    
    # Process each image
    for i, img_path in enumerate(sample_images):
        try:
            print(f"Processing {i+1}/101: {img_path.name}")
            
            # Load original image
            orig = cv2.imread(str(img_path), 0)
            if orig is None:
                print(f"Warning: could not read {img_path}, skipping.")
                continue
            
            # Embed watermark
            wm = embedding(str(img_path), mark_file)
            v_wm = _extract_ratio_vector(wm, orig)
            
            # Positives: light attacks (watermark should survive)
            light_attacks = [
                ("jpeg", {"qf": 90}),
                ("blur", {"ksize": 3}),
                ("resize", {"scale": 0.9}),
            ]
            
            for name, params in light_attacks:
                if name == "jpeg":
                    att = A.attack_jpeg(wm, **params)
                elif name == "blur":
                    att = A.attack_blur(wm, **params)
                elif name == "resize":
                    att = A.attack_resize(wm, **params)
                
                v_att = _extract_ratio_vector(att, orig)
                sim = _similarity(v_wm, v_att)
                scores.append(sim)
                labels.append(1)  # Positive (watermark present)
            
            # Negatives: clean image and random vector (no watermark)
            v_clean = _extract_ratio_vector(orig, orig)
            sim_clean = _similarity(v_wm, v_clean)
            scores.append(sim_clean)
            labels.append(0)  # Negative (no watermark)
            
            # Random vector
            rand = np.random.randn(v_wm.size).astype(np.float32)
            sim_rand = _similarity(v_wm, rand)
            scores.append(sim_rand)
            labels.append(0)  # Negative (no watermark)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Convert to numpy arrays
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Collected {len(scores)} samples: {np.sum(labels)} positives, {len(labels) - np.sum(labels)} negatives")
    
    # Compute ROC curve
    fpr, tpr, thresholds = compute_roc_curve(labels, scores)
    auc = compute_auc(fpr, tpr)
    
    # Select optimal threshold (FPR <= 0.1, maximize TPR)
    valid_indices = np.where(fpr <= 0.1)[0]
    if len(valid_indices) > 0:
        optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
    else:
        optimal_idx = np.argmin(fpr)
    
    tau = float(thresholds[optimal_idx])
    optimal_fpr = float(fpr[optimal_idx])
    optimal_tpr = float(tpr[optimal_idx])
    
    # Save results
    results = {
        "tau": tau,
        "AUC": float(auc),
        "FPR": optimal_fpr,
        "TPR": optimal_tpr,
        "total_samples": len(scores),
        "positives": int(np.sum(labels)),
        "negatives": int(len(labels) - np.sum(labels))
    }
    
    with open("tau.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nROC completed on {len(sample_images)} images.")
    print(f"AUC={auc:.3f}  tau={tau:.6f}  @ FPR={optimal_fpr:.3f}, TPR={optimal_tpr:.3f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, color='navy', label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=8, label=f"Optimal τ={tau:.6f}")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({len(sample_images)} Images)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc.png", dpi=300)
    plt.savefig("roc.pdf")
    plt.close()
    
    print("ROC plot saved as roc.png and roc.pdf")
    
    return results

if __name__ == "__main__":
    results = main()

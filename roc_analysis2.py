import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def estimate_threshold_roc():
    """
    ROC analysis code to estimate optimal threshold
    Run this BEFORE the competition!
    """
    # Test images for ROC analysis
    test_images = [
        'test_image1.bmp',
        'test_image2.bmp', 
        'test_image3.bmp'
    ]
    
    watermark_path = 'watermark.npy'
    
    scores = []
    labels = []
    
    # Attack variations for testing
    attack_variations = [
        ('jpeg', [90]), ('jpeg', [70]), ('jpeg', [50]), ('jpeg', [30]),
        ('awgn', [5]), ('awgn', [15]), ('awgn', [25]), ('awgn', [35]),
        ('blur', [3]), ('blur', [5]), ('blur', [7]),
        ('median', [3]), ('median', [5]), ('median', [7]),
        ('sharp', [1.0, 0.3]), ('sharp', [1.5, 0.5]),
        ('resize', [0.8]), ('resize', [0.6]), ('resize', [0.5])
    ]
    
    print("Starting ROC analysis...")
    
    for img_idx, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
            
        print(f"Processing {img_path}...")
        
        # Embed watermark
        try:
            from embedding import embedding
            watermarked_img = embedding(img_path, watermark_path)
            watermarked_path = f"temp_watermarked_{img_idx}.bmp"
            Image.fromarray(watermarked_img).save(watermarked_path)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Test attacks (TRUE POSITIVES)
        for attack_name, params in attack_variations:
            try:
                from attacks import attacks
                attacked_img = attacks(watermarked_path, attack_name, params)
                attacked_path = f"temp_attacked_{img_idx}.bmp"
                Image.fromarray(attacked_img).save(attacked_path)
                
                similarity = extract_similarity(img_path, watermarked_path, attacked_path)
                scores.append(similarity)
                labels.append(1)
                
                if os.path.exists(attacked_path):
                    os.remove(attacked_path)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Generate FALSE POSITIVES
        for fp_idx in range(10):
            try:
                random_watermark = np.random.randint(0, 2, 1024)
                random_watermark_path = f"temp_random_{fp_idx}.npy"
                np.save(random_watermark_path, random_watermark)
                
                random_watermarked = embedding(img_path, random_watermark_path)
                random_watermarked_path = f"temp_random_watermarked_{fp_idx}.bmp"
                Image.fromarray(random_watermarked).save(random_watermarked_path)
                
                mild_attacked = attacks(random_watermarked_path, 'jpeg', [80])
                mild_attacked_path = f"temp_mild_{fp_idx}.bmp"
                Image.fromarray(mild_attacked).save(mild_attacked_path)
                
                false_similarity = extract_similarity(img_path, watermarked_path, mild_attacked_path)
                scores.append(false_similarity)
                labels.append(0)
                
                # Cleanup
                for temp_file in [random_watermark_path, random_watermarked_path, mild_attacked_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if os.path.exists(watermarked_path):
            os.remove(watermarked_path)
    
    if len(scores) < 20:
        print("Insufficient data. Using default threshold.")
        return 0.7
    
    # Compute ROC
    scores = np.array(scores)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold for FPR <= 0.1
    target_fpr = 0.1
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(tpr[valid_indices])]
        optimal_threshold = thresholds[best_idx]
    else:
        best_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[best_idx]
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Watermark Detection')
    plt.axvline(x=target_fpr, color='red', linestyle=':', label=f'Target FPR={target_fpr}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return optimal_threshold


def extract_similarity(original_path, watermarked_path, attacked_path):
    """Helper to extract similarity"""
    try:
        from detection import extract_watermark_svd, compute_similarity
        
        original = np.array(Image.open(original_path).convert('L')).astype(np.float32)
        watermarked = np.array(Image.open(watermarked_path).convert('L')).astype(np.float32)
        attacked = np.array(Image.open(attacked_path).convert('L')).astype(np.float32)
        
        w1 = extract_watermark_svd(original, watermarked)
        w2 = extract_watermark_svd(original, attacked)
        
        return compute_similarity(w1, w2)
    except:
        return 0.0


if __name__ == "__main__":
    print("ROC Analysis for Threshold Estimation")
    optimal_threshold = estimate_threshold_roc()
    print(f"\nRECOMMENDED THRESHOLD: {optimal_threshold:.4f}")
    print("Update this in detection.py!")

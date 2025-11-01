import os, glob, cv2, json
import numpy as np
from embedding import embedding  # Import the watermark embedding function
from detection_shadowmark import detection # Import the watermark detection function

# Define the search pattern for our source images (all BMPs in the 'images' folder)
IMG_GLOB = os.path.join('images', '*.bmp')
# Find and sort all image paths that match the pattern
imgs = sorted(glob.glob(IMG_GLOB))
# Stop the script if no images were found, as the test can't run
assert imgs, f"No BMP images found in {IMG_GLOB}"

# --- Experiment Configuration ---
trials = 5  # Number of different random watermarks to test
fprs = []   # A list to store the False Positive Rate (FPR) from each trial

# --- Main loop: run the test once for each (different) watermark key ---
for t in range(trials):
    
    # --- 1. Generate a new, random watermark for this trial ---
    
    # Create a seeded random number generator (RNG) for this trial
    # Using '1000 + t' ensures each trial's watermark is different, but reproducible
    rng = np.random.default_rng(1000 + t)
    # Generate a random binary watermark (1024 bits)
    watermark = rng.integers(0, 2, 1024, dtype=np.uint8)
    # Define a temporary file path to save this trial's watermark key
    key_path = f'tmp_watermark_{t}.npy'
    # Save the watermark to disk so the 'embedding' function can read it
    np.save(key_path, watermark)

    # Reset counters for this specific trial
    total = 0 # Total number of images processed
    fp = 0    # Number of false positives found

    # --- 2. Inner loop: test every image against the current watermark ---
    for p in imgs:
        
        # --- This is the setup for the detection function ---
        # The detector needs: 1. original, 2. watermarked
        # So, we must create a watermarked version *only* to serve as the 'input2' reference.
        
        # Embed the current trial's watermark (from key_path) into the image 'p'
        wm_img = embedding(p, key_path)
        # Save this watermarked image to a temporary file
        cv2.imwrite('temp_watermarked.bmp', wm_img)

        # --- 3. Run the FPR test ---
        # The core of the False Positive test is to run detection ON THE ORIGINAL IMAGE.
        #   input1 (original):   p
        #   input2 (watermarked): 'temp_watermarked.bmp' (the reference)
        #   input3 (test image): p  <-- We are testing the ORIGINAL image
        #
        # We are asking: "Is the watermark (defined by p and temp_watermarked.bmp)
        # present in the *original* image 'p'?"
        # The correct answer should always be NO.
        detected, _ = detection(p, 'temp_watermarked.bmp', p)
        
        total += 1 # Increment the total number of tests
        
        # If the detector *claims* the watermark is present (detected=1),
        # it has made a mistake. This is a False Positive.
        if detected:
            fp += 1
            
    # --- 4. Calculate and store this trial's results ---
    
    # Calculate the False Positive Rate for this specific watermark key
    fpr = fp / total
    # Store this trial's FPR in our list
    fprs.append(fpr)
    print(f"[trial {t}] FPR = {fp}/{total} = {fpr:.3%}")

# --- 5. Final Summary ---
# After all trials are done, print a summary of the FPRs found
print("\nSummary over 5 random watermarks:")
print(f"min FPR: {min(fprs):.3%}") # The best-case (lowest) FPR
print(f"avg FPR: {np.mean(fprs):.3%}") # The average FPR across all keys
print(f"max FPR: {max(fprs):.3%}") # The worst-case (highest) FPR

os.remove('temp_watermarked.bmp')
os.remove(key_path)
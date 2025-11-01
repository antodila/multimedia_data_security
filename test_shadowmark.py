import os
import cv2
import glob
import numpy as np
from embedding import embedding  # Import the watermark embedding function
from detection_shadowmark import detection  # Import the watermark detection function

# -----------------------------------------------------------------
# 1. Setup: Generate a single random watermark for this test run
# -----------------------------------------------------------------

# 'genera watermark casuale 1024 bit'
# Generate a random 1024-bit binary watermark (0s and 1s)
watermark = np.random.randint(0, 2, 1024)
# Save this watermark to a temporary file
np.save('tmp_watermark.npy', watermark)
# Store the file path in a constant for easy access
WATERMARK_KEY_PATH = 'tmp_watermark.npy'

# -----------------------------------------------------------------
# 2. Find all images to be tested
# -----------------------------------------------------------------

# 'cartella immagini: ./images/*.bmp'
# Create a search pattern for all .bmp files in the 'images' folder
test_images = glob.glob(os.path.join('images', '*.bmp'))

# -----------------------------------------------------------------
# 3. Initialize counters for the FPR test
# -----------------------------------------------------------------
total_count = 0     # Total number of detection tests performed
problem_count = 0   # Count of False Positives (when the watermark is found incorrectly)

# -----------------------------------------------------------------
# 4. Loop through each image and perform the FPR test
# -----------------------------------------------------------------

# Iterate over every image path found in the folder
for img_path in test_images:
    print(f'Testing {img_path}:')

    # --- EMBEDDING ---
    # We must create a watermarked version of the image.
    # The detector's API (input2) requires this as the "ground truth" reference,
    # even though we won't be testing this file itself.
    watermarked_image = embedding(img_path, WATERMARK_KEY_PATH)
    
    # Define a temporary path to save this watermarked reference image
    watermarked_path = 'temp_watermarked.bmp'
    # Write the watermarked image to disk
    cv2.imwrite(watermarked_path, watermarked_image)

    # --- DETECTION on ORIGINAL image (this measures FPR) ---
    # This is the core of the False Positive Rate test.
    # We ask the detector to find the watermark in the *ORIGINAL* (unwatermarked) image.
    #
    #   input1 (original):   img_path
    #   input2 (watermarked): watermarked_path (the reference)
    #   input3 (test image): img_path  <-- We are testing the ORIGINAL image
    #
    # A correct result here is 'detected_orig = 0'.
    detected_orig, wpsnr_orig = detection(img_path, watermarked_path, img_path)
    
    # If 'detected_orig' is 1, it means a watermark was found where none existed.
    if detected_orig:
        print('  - (ERROR) Watermark incorrectly found in original image.')

    # --- Update counters ---
    total_count += 1  # Increment the total number of images tested
    if detected_orig:
        problem_count += 1 # Increment the error counter

# -----------------------------------------------------------------
# 5. Print the final FPR result
# -----------------------------------------------------------------
print(f'FPR: {problem_count}/{total_count} ({(problem_count/total_count)*100:.2f}%)')

try:
    os.remove('temp_watermarked.bmp')
    os.remove('tmp_watermark.npy')
except FileNotFoundError:
    pass
import os
import numpy as np
from scipy.fft import dct, idct
import cv2

# === Parameter Configuration ===
IMG_PATH = 'lena_grey.bmp'  # Use the image copied into the project folder
MARK_SIZE = 1024
ALPHA = 0.1
MODE = 'multiplicative'  # 'additive' or 'multiplicative'
SEED = 123

np.random.seed(SEED)

# === Read the Grayscale Image ===
image = cv2.imread(IMG_PATH, 0)
if image is None:
    raise FileNotFoundError(f"Could not read image at path: {IMG_PATH}")

# === Embedding Function ===
def embedding(image, mark_size, alpha, v='multiplicative'):
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct, axis=None)
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]

    # Generate a random watermark (binary)
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)

    watermarked_dct = ori_dct.copy()
    for idx, (loc, mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + (alpha * mark_val)

    watermarked_dct *= sign
    watermarked = np.uint8(idct(idct(watermarked_dct, axis=1, norm='ortho'), axis=0, norm='ortho'))
    return mark, watermarked

# === Embed the watermark and save results ===
mark, watermarked = embedding(image, MARK_SIZE, ALPHA, MODE)
cv2.imwrite('watermarked.bmp', watermarked)
print('Watermarked image saved as watermarked.bmp')
print('Watermark sequence saved as mark.npy')

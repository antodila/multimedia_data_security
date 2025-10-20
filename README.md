Spread Spectrum Watermark Embedder — Complete Guide

This repository implements a DCT-based Spread Spectrum watermark embedding pipeline and the supporting tools needed for the Multimedia Data Security challenge: embedding, detection (non-blind), attacks, smoke tests, ROC thresholding and an automatic attacker grid-search.

Status: embedding works, detection uses a residual (ratio) feature, ROC computed and tau selection workflow available.
Replace groupname with your actual group id when saving/submitting images.

Project layout (files you should have)

embed_spread_spectrum.py — main embedder: embedding(input_image_path, watermark_path) → returns watermarked_image_array

detection_group.py — detection implementation (non-blind). Contains detection(input1,input2,input3) and helper _extract_ratio_vector(...). Set tau (numeric) after ROC. Rename to detection_<groupname>.py for final submission.

attacks.py — permitted attacks: attack_jpeg, attack_awgn, attack_blur, attack_median, attack_resize, attack_sharp.

test_embed.py — quick script that calls embedder and saves groupname_lena_grey.bmp.

test_smoke.py — smoke tests: TP, TN, and quick attack checks.

roc_threshold.py — builds ROC from scores and writes tau.json (selects τ with FPR ≤ 0.1).

attack_and_detect.py — example: apply an attack (e.g. resize 0.6), save attacked image and run detection.

grid_search_attacks.py — (optional) grid search attacker, finds best valid attack per image and writes attack_log.csv.

lena_grey.bmp — sample grayscale image (replaceable).

mark.npy — watermark (1024 bits). If not provided by challenge, generate a dummy for development.

requirements.txt — Python dependencies.

.gitignore — ignore your virtual env, etc.

Requirements / virtual environment

Create and activate a venv, then install dependencies:

Windows (PowerShell):

python -m venv OFF_MDS_ENV
OFF_MDS_ENV\Scripts\Activate.ps1
pip install -r requirements.txt


macOS / Linux:

python3 -m venv OFF_MDS_ENV
source OFF_MDS_ENV/bin/activate
pip install -r requirements.txt


If you don't have requirements.txt or need extra packages:

pip install numpy scipy opencv-python scikit-learn pandas matplotlib

Quick start — run the embedder (generate a watermarked image)

Make sure lena_grey.bmp and mark.npy exist. If you do not have a mark.npy:

python - <<'PY'
import numpy as np
np.save('mark.npy', (np.random.rand(1024) > 0.5).astype('uint8'))
print("dummy mark.npy created")
PY


Run the test embed script (this saves groupname_lena_grey.bmp — change groupname inside test_embed.py before running):

python -X dev -u test_embed.py
# Expected: "Watermarked image saved as groupname_lena_grey.bmp"


Or call the embedder directly in a Python session:

from embed_spread_spectrum import embedding
import cv2
wm = embedding("lena_grey.bmp", "mark.npy")   # returns uint8 array
cv2.imwrite("groupname_lena_grey.bmp", wm)

Smoke tests (3 quick checks)

Run the smoke test to verify:

True positive (watermarked recognized)

True negative (clean image recognized as not watermarked)

A few attack examples show similarity drop and report WPSNR

python -X dev -u test_smoke.py


Interpret outputs:

TP self-sim should be ≈ 1.0

TN sim (wm vs clean) should be low (≈ 0)

The printed attack results show sim and WPSNR for each tested attack. Look for a candidate where sim < eventual tau and WPSNR >= 35 dB.

Compute ROC and choose threshold τ

Collect many similarity scores (positives vs negatives) and compute ROC. The provided script selects τ with FPR ≤ 0.1 and maximal TPR among those:

pip install scikit-learn
python -X dev -u roc_threshold.py


Script prints something like:

AUC=0.995  tau=0.307841  @ FPR=0.000, TPR=1.000


roc_threshold.py creates tau.json. To display it in Git Bash:

cat tau.json


Then edit detection_group.py and set:

tau = 0.307841   # replace with the number from tau.json


Save and, for the final submission, rename the file to:

detection_<yourgroup>.py


Important: detection must include a literal numeric tau and must not compute ROC at runtime.

Run full detection & an example attack

Create an attacked version by running attack_and_detect.py (it uses the attacks.py functions and your watermarked file):

python attack_and_detect.py
# Expected output example: presence: 0 wpsnr: 35.83456802368164


Meaning: detection returned 0 (watermark removed) and WPSNR >= 35 dB (attack is valid per rules).

If you need other attack parameters, edit attack_and_detect.py or call attacks.py functions in a small script.

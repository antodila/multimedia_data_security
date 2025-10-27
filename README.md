# Shadowmark — Multimedia Data Security (2025)

This repository contains the code used for the Multimedia Data Security competition (University of Trento, 2025). The project implements a DCT‑domain spread‑spectrum watermarking system with the required components: embedding, detection, ROC thresholding and attack modules.

Team: **shadowmark**

## Project layout

Main files:

- `embedding.py` — public API: `embedding(input_image_path, watermark_path_or_array) -> np.uint8`.
- `detection_shadowmark.py` — public API: `detection(original, watermarked, attacked) -> (presence:int, wpsnr:float)`.
	- Note: the detector will automatically load `tau` from `tau.json` when present; a fallback value is kept otherwise.
- `attacks.py` — implements the permitted attacks: `attack_jpeg`, `attack_awgn`, `attack_blur`, `attack_median`, `attack_resize`, `attack_sharp`.
- `roc_threshold.py` — computes ROC on the provided dataset, selects `tau` and saves `tau.json`, `roc.png`, `roc.pdf`.
- `test_smoke.py` — quick sanity checks (TP, TN and a few attacks).
- `mark.npy` — watermark array (1024 bits). Generate if not provided.

Useful folders:

- `images/` — images used for ROC (expected names: `0000.bmp` ... `0100.bmp`).

## Quick setup (Windows / Git Bash)

1. Create and activate a virtual environment (Git Bash / MinGW on Windows):

```bash
python -m venv OFF_MDS_ENV
source OFF_MDS_ENV/Scripts/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Quick check:

```bash
python -c "import numpy, cv2, scipy; print('libs ok')"
```

## Quick start — common commands

1) Generate a test watermark (if you don't have `mark.npy`):

```bash
python - <<'PY'
import numpy as np
np.save('mark.npy', (np.random.rand(1024) > 0.5).astype('uint8'))
print('mark.npy created')
PY
```

2) Embed the watermark into the sample image (`lena_grey.bmp`):

```bash
python - <<'PY'
from embedding import embedding
import cv2
wm = embedding('lena_grey.bmp', 'mark.npy')
cv2.imwrite('shadowmark_lena_grey.bmp', wm)
print('Saved: shadowmark_lena_grey.bmp')
PY
```

3) Compute ROC and save `tau.json` (run on the `images/` dataset):

```bash
python roc_threshold.py
```

4) Run the detector (it uses `tau.json` if present):

```bash
python - <<'PY'
from detection_shadowmark import detection
print(detection('lena_grey.bmp', 'shadowmark_lena_grey.bmp', 'shadowmark_lena_grey.bmp'))
PY
```

5) Apply a test attack and check detection:

```bash
python - <<'PY'
import cv2
from attacks import attack_resize
from detection_shadowmark import detection
wm = cv2.imread('shadowmark_lena_grey.bmp', 0)
att = attack_resize(wm, scale=0.6)
cv2.imwrite('shadowmark_lena_grey_resize06.bmp', att)
print(detection('lena_grey.bmp', 'shadowmark_lena_grey.bmp', 'shadowmark_lena_grey_resize06.bmp'))
PY
```

## Recommended tests

- `python test_smoke.py` — quick sanity checks (TP/TN and sample attacks).
- Measure detection runtime to ensure it meets the competition limit (< 5 seconds):

```bash
python - <<'PY'
import time
from detection_shadowmark import detection
t0 = time.time()
print(detection('lena_grey.bmp','shadowmark_lena_grey.bmp','shadowmark_lena_grey.bmp'))
print('elapsed:', time.time() - t0)
PY
```

## Notes and best practices

- `detection_shadowmark.py` does not print during normal execution (requirement of the competition).
- `tau` is computed offline using `roc_threshold.py` and saved to `tau.json`; the detector will load this file automatically if present. If you prefer not to include `tau.json` in the submission, regenerate it before the competition.
- For an attack to be considered successful in the competition the conditions are:
	- `presence == 0` (watermark not detected) AND
	- `wpsnr >= 35` dB (image quality preserved).

## Submission checklist (before the deadline)

1. Ensure required files are present and up to date:
	 - `embedding.py`, `detection_shadowmark.py`, `attacks.py`, `roc_threshold.py`, and either `tau.json` or a script to regenerate it.
2. Run `python test_smoke.py` and verify expected outputs.
3. Verify detection runtime is < 5 seconds on the target machine.
4. Prepare `attack_log.csv` listing successful attacks (columns: Image, Group, WPSNR, Attack, Params, OutputFile).
5. Commit and push (or open a PR) your final code.

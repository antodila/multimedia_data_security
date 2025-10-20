# Spread Spectrum Watermark Embedder

This project provides a simple implementation of image watermark embedding using the robust Spread Spectrum technique, based on DCT transforms.

## Features
- Embeds a random binary watermark in a grayscale image using spread spectrum (multiplicative/additive) in the DCT domain
- Saves both the watermarked image and the watermark

## Project Structure
- `embed_spread_spectrum.py`  — script for embedding a watermark
- `lena_grey.bmp`             — sample grayscale input image (replaceable)
- `watermarked.bmp`           — the generated watermarked image
- `mark.npy`                  — the generated binary watermark (NumPy array)
- `requirements.txt`          — required pip dependencies
- `.gitignore`                — ensures the virtual environment is not committed

## Setup Instructions

### 1. Clone the repository and enter the project folder
```
cd path/to/your/MultiMediaDataSecurity/project
```

### 2. Create and activate a Python virtual environment

#### On Windows:
```
python -m venv OFF_MDS_ENV
OFF_MDS_ENV\Scripts\activate
```

#### On macOS/Linux/Unix:
```
python3 -m venv OFF_MDS_ENV
source OFF_MDS_ENV/bin/activate
```

### 3. Install required dependencies
```
pip install -r requirements.txt
```

### 4. Run the embedding script
```
python embed_spread_spectrum.py
```

- The script uses `lena_grey.bmp` by default (must be present in the `project` directory)
- Outputs will be `watermarked.bmp` and `mark.npy` in the same folder

---

## Customization
- To watermark your own image: replace `lena_grey.bmp` and/or adjust `IMG_PATH` at the top of `embed_spread_spectrum.py`.
- Adjust watermark length, `alpha`, or mode (`multiplicative` or `additive`) in the script if needed.

---

## Notes
- The virtual environment (`OFF_MDS_ENV`) is ignored by git and should not be committed.
- To install dependencies elsewhere, use the provided `requirements.txt`.

---

## License
This project is for educational purposes.

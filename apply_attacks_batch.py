#!/usr/bin/env python3
"""
apply_attacks_batch.py
Apply a set of attacks to downloaded watermarked images and save outputs
with the required competition naming convention: attacker_victim_imageName.bmp

Usage:
    python apply_attacks_batch.py

Configure the ATTACKS list below. Put downloaded watermarked images (from other groups)
in the current folder or in the DOWNLOAD_DIR directory.

The script will:
 - look for files matching "*_*.bmp" and treat the first token before the underscore as the victim/group
 - skip files already produced by our group (prefix "shadowmark_")
 - produce attacked files in OUT_DIR with names: shadowmark_<victim>_<imageName>.bmp
 - generate a CSV log: attack_batch_log.csv
"""
import os
import glob
import csv
import cv2
import numpy as np
import attacks as A

# -----------------------
# Configuration
# -----------------------
ATTACKER = "shadowmark"       # your group name
DOWNLOAD_DIR = "."            # where the downloaded watermarked files are placed
OUT_DIR = "attacked_for_submission"
LOG_CSV = "attack_batch_log.csv"

# Define your attack candidates here.
# Each entry: (attack_name_for_log, callable_that_accepts_numpy_image, params_string)
ATTACKS = [
    ("jpeg40",         lambda img: A.attack_jpeg(img, 40),                         "qf=40"),
    ("jpeg60",         lambda img: A.attack_jpeg(img, 60),                         "qf=60"),
    ("blur3+j60",      lambda img: A.attack_jpeg(A.attack_blur(img,3), 60),        "ksize=3,qf=60"),
    ("blur3+j60+med3", lambda img: A.attack_blur_jpeg_median(img, 3, 60, 3),       "ksize=3,qf=60,med=3"),
    ("resize06+j60",   lambda img: A.attack_resize_jpeg_median(img, 0.6, 60, 3),   "scale=0.6,qf=60,med=3"),
    ("blur5+j50+med3", lambda img: A.attack_blur_jpeg_median(img, 5, 50, 3),       "ksize=5,qf=50,med=3"),
    ("resize08+j60+med3", lambda img: A.attack_resize_jpeg_median(img, 0.8, 60, 3),"scale=0.8,qf=60,med=3"),
]


# -----------------------
# Helpers
# -----------------------
def safe_output_path(attacker, victim, orig_basename, out_dir, suffix=None):
    """
    Build 'attacker_victim_origBasename' and ensure we don't overwrite files.
    orig_basename is the image filename with extension (e.g. '0000.bmp' or 'image.bmp').
    """
    name_noext, ext = os.path.splitext(orig_basename)
    if suffix:
        name_noext = f"{name_noext}{suffix}"
    out_name = f"{attacker}_{victim}_{name_noext}{ext}"
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        i = 1
        while True:
            alt = os.path.join(out_dir, f"{attacker}_{victim}_{name_noext}__{i}{ext}")
            if not os.path.exists(alt):
                out_path = alt
                break
            i += 1
    return out_path

def parse_victim_and_image(filename):
    """
    Given a filename like 'groupB_0000.bmp' or 'groupB_some_name.bmp',
    return (victim_group, image_basename). If filename doesn't match, returns (None, None).
    """
    base = os.path.basename(filename)
    if "_" not in base:
        return None, None
    parts = base.split("_", 1)
    victim = parts[0]
    image_name = parts[1]
    return victim, image_name

# -----------------------
# Main
# -----------------------
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # find candidate files: any "*_*.bmp" inside DOWNLOAD_DIR
    pattern = os.path.join(DOWNLOAD_DIR, "*_*.bmp")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No downloaded watermarked files found (pattern: *_*.bmp). Put victim files in the download dir and retry.")
        return

    # Filter: skip files that are already outputs from our group (prefix ATTACKER_)
    victim_files = [f for f in files if not os.path.basename(f).startswith(f"{ATTACKER}_")]

    if not victim_files:
        print("No victim files to attack (all found files have your attacker prefix).")
        return

    # Prepare CSV log
    csv_fields = ["attacker", "victim", "original_wm", "attacked_out", "attack_name", "params"]
    with open(LOG_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_fields)

        # iterate victims
        for vf in victim_files:
            victim, image_basename = parse_victim_and_image(vf)
            if victim is None:
                print(f"[WARN] skipping file with unexpected name: {vf}")
                continue

            # load the watermarked image (grayscale)
            img = cv2.imread(vf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] cannot read {vf}, skipping")
                continue

            # apply each attack and save
            for atk_name, atk_fn, params in ATTACKS:
                try:
                    attacked = atk_fn(img)  # returns numpy uint8 image
                except Exception as e:
                    print(f"[ERROR] attack {atk_name} failed on {vf}: {e}")
                    continue

                out_path = safe_output_path(ATTACKER, victim, image_basename, OUT_DIR)
                ok = cv2.imwrite(out_path, attacked)
                if not ok:
                    print(f"[ERROR] failed to write {out_path}")
                    continue

                writer.writerow([ATTACKER, victim, os.path.basename(vf), os.path.basename(out_path), atk_name, params])
                print(f"Saved: {out_path}  (attack={atk_name} params={params})")

    print(f"\nDone. Attacked images in '{OUT_DIR}'. CSV log: {LOG_CSV}")

if __name__ == "__main__":
    run()

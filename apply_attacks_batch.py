#!/usr/bin/env python3
"""
apply_attacks_batch.py
Apply a set of attacks to downloaded watermarked images and save outputs
with the required competition naming convention: attacker_victim_imageName.bmp

Usage:
    python apply_attacks_batch.py
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
DOWNLOAD_DIR = "download_test"            # where the downloaded watermarked files are placed
OUT_DIR = "attacked_for_submission"
LOG_CSV = "attack_batch_log.csv"

# --- ATTACKS ---
# FORMAT: (log_name, attack_name_in_dispatcher, parameter_list, log_params_string)
ATTACKS = [
    # High WPSNR attacks
    ("jpeg_60", "jpeg", [60], "QF=60"),
    ("jpeg_50", "jpeg", [50], "QF=50"),
    ("jpeg_40", "jpeg", [40], "QF=40"),
    ("resize_09", "resize", [0.9], "scale=0.9"),
    ("resize_08", "resize", [0.8], "scale=0.8"),
    
    # Combined attacks
    ("strategy_3_smart", "strategy_3_smart", None, "Strategy 3: Smart (Blur+JPEG+Median)"),
    ("strategy_4_chaos", "strategy_4_chaos", None, "Strategy 4: Chaos"),
    ("strategy_9_surgical", "strategy_9_surgical", None, "Strategy 9: Surgical"),
    ("strategy_1_stealth", "strategy_1_stealth", None, "Strategy 1: Stealth"),
    ("strategy_2_brutal", "strategy_2_brutal", None, "s2: blur15+resize03+awgn20"),
    ("strategy_5_precision", "strategy_5_precision", None, "s5: blur11+resize08+awgn15"),
    ("strategy_6_wave", "strategy_6_wave", None, "s6: multi-stage mild"),
    ("strategy_7_counter", "strategy_7_counter", None, "s7: jpeg80+blur13+resize09"),
    ("strategy_8_freq", "strategy_8_freq", None, "s8: progressive blur+awgn6+med3"),
    ("ULTIMATE_COMBO", "ULTIMATE_COMBO", None, "jpeg5+awgn30+resize03+med5"),
    
    # Medium attacks
    ("median_3", "median", [3], "Median Filter k=3"),
    ("median_5", "median", [5], "Median Filter k=5"),
    ("blur_5", "blur", [5], "Gaussian Blur k=5"),
    ("blur_7", "blur", [7], "Gaussian Blur k=7"),
    ("sharp_10", "sharp", [1.0], "Sharpening amount=1.0"),
    ("sharp_15", "sharp", [1.5], "Sharpening amount=1.5"),
    ("awgn_10", "awgn", [10], "AWGN sigma=10"),
    ("awgn_15", "awgn", [15], "AWGN sigma=15"),
]

# -----------------------
# Helpers
# -----------------------
def safe_output_path(attacker, victim, orig_basename, out_dir, suffix=None):
    """
    Build 'attacker_victim_origBasename' and ensure we don't overwrite files.
    """
    name_noext, ext = os.path.splitext(orig_basename)
    # Use the log_name (suffix) to make the filename unique
    out_name = f"{attacker}_{victim}_{name_noext}_{suffix}{ext}"
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        i = 1
        while True:
            alt = os.path.join(out_dir, f"{attacker}_{victim}_{name_noext}_{suffix}__{i}{ext}")
            if not os.path.exists(alt):
                out_path = alt
                break
            i += 1
    return out_path

def parse_victim_and_image(filename):
    """
    Given a filename like 'groupB_0000.bmp',
    return (victim_group, image_basename).
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

    pattern = os.path.join(DOWNLOAD_DIR, "*_*.bmp")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No downloaded watermarked files found (pattern: *_*.bmp). Put victim files in the download dir and retry.")
        return

    victim_files = [f for f in files if not os.path.basename(f).startswith(f"{ATTACKER}_")]

    if not victim_files:
        print("No victim files to attack (all found files have your attacker prefix).")
        return

    csv_fields = ["attacker", "victim", "original_wm", "attacked_out", "attack_name", "params"]
    with open(LOG_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_fields)

        for vf in victim_files:
            victim, image_basename = parse_victim_and_image(vf)
            if victim is None:
                print(f"[WARN] skipping file with unexpected name: {vf}")
                continue

            # load the watermarked image (grayscale)
            # We don't need to do this. The attacks() function reads the path.
            
            # apply each attack and save
            # THIS IS THE FIXED LOOP
            for log_name, attack_name, param_array, params_str in ATTACKS:
                try:
                    # Call the main dispatcher function A.attacks()
                    # It reads the file 'vf' and applies the attack
                    attacked = A.attacks(vf, attack_name, param_array)
                    
                except Exception as e:
                    print(f"[ERROR] attack {attack_name} failed on {vf}: {e}")
                    continue

                # Pass log_name as the suffix to create a unique filename
                out_path = safe_output_path(ATTACKER, victim, image_basename, OUT_DIR, suffix=log_name)
                ok = cv2.imwrite(out_path, attacked)
                if not ok:
                    print(f"[ERROR] failed to write {out_path}")
                    continue

                writer.writerow([ATTACKER, victim, os.path.basename(vf), os.path.basename(out_path), log_name, params_str])
                print(f"Saved: {out_path}  (attack={log_name} params={params_str})")

    print(f"\nDone. Attacked images in '{OUT_DIR}'. CSV log: {LOG_CSV}")

if __name__ == "__main__":
    run()
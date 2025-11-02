#!/usr/bin/env python3
# evaluate_from_log.py
import os, csv
from collections import defaultdict
import cv2
from detection_shadowmark import detection

# Config: adegua solo se hai cartelle diverse
IMAGES_DIR   = "images"
DOWNLOAD_DIR = "download_test"
OUT_DIR      = "attacked_for_submission"
LOG_IN       = "attack_batch_log.csv"
LOG_OUT      = "attack_batch_checked.csv"

def path_original(image_basename):
    # es: "0093.bmp" -> "images/0093.bmp"
    return os.path.join(IMAGES_DIR, image_basename)

def path_watermarked(victim, image_basename):
    # es: victimA + "0093.bmp" -> "download_test/victimA_0093.bmp"
    return os.path.join(DOWNLOAD_DIR, f"{victim}_{image_basename}")

def path_attacked(attacker, victim, attacked_out):
    # attacked_out è già un basename tipo "shadowmark_victimA_0093__2.bmp"
    return os.path.join(OUT_DIR, attacked_out)

def run():
    assert os.path.exists(LOG_IN), f"Missing {LOG_IN}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Statistiche per attack_name
    stats = defaultdict(lambda: {"tot":0, "succ":0, "wps_sum":0.0, "wps_min":999.0})

    with open(LOG_IN, newline="") as fin, open(LOG_OUT, "w", newline="") as fout:
        rd = csv.DictReader(fin)
        wr = csv.writer(fout)
        wr.writerow(["victim","image","attack_name","params","presence","wpsnr","valid","attacked_path"])

        for row in rd:
            attacker   = row["attacker"]             # "shadowmark"
            victim     = row["victim"]               # "victimA"
            orig_wm    = row["original_wm"]          # es "victimA_0093.bmp"
            attacked   = row["attacked_out"]         # es "shadowmark_victimA_0093__2.bmp"
            attackname = row["attack_name"]
            params     = row.get("params","")

            # ricostruisci image_basename: "victimA_0093.bmp" -> "0093.bmp"
            try:
                image_basename = orig_wm.split("_", 1)[1]
            except Exception:
                print(f"[WARN] nome inatteso original_wm={orig_wm}; salto")
                continue

            p_orig = path_original(image_basename)
            p_wm   = path_watermarked(victim, image_basename)
            p_att  = path_attacked(attacker, victim, attacked)

            # check esistenza file minimi
            missing = [p for p in [p_orig, p_wm, p_att] if not os.path.exists(p)]
            if missing:
                print(f"[WARN] file mancanti per {attacked}: {missing}")
                continue

            try:
                presence, wps = detection(p_orig, p_wm, p_att)
                wps = float(wps)
            except Exception as e:
                print(f"[ERROR] detection fallita su {p_att}: {e}")
                continue

            valid = int(presence == 0 and wps >= 35.0)

            # log riga
            wr.writerow([victim, image_basename, attackname, params, presence, f"{wps:.2f}", valid, p_att])

            # stats
            s = stats[attackname]
            s["tot"] += 1
            if valid:
                s["succ"] += 1
                s["wps_sum"] += wps
                if wps < s["wps_min"]:
                    s["wps_min"] = wps

    # stampa summary
    print("\n== SUMMARY by attack_name ==")
    for name, s in sorted(stats.items()):
        tot = s["tot"]; succ = s["succ"]
        rate = (succ / tot) if tot else 0.0
        avgw = (s["wps_sum"]/succ) if succ else 0.0
        minw = (s["wps_min"] if succ else 0.0)
        print(f"{name:20} success={succ:4d}/{tot:4d}  rate={rate:6.2%}  avgWPSNR={avgw:6.2f}  minWPSNR={minw:6.2f}")
    print(f"\nRisultati riga-per-riga in: {LOG_OUT}")

if __name__ == "__main__":
    run()

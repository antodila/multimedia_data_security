import os, csv
import cv2
from detection_shadowmark import detection
import attacks as A

ORIG = "lena_grey.bmp"
WM   = "shadowmark_lena_grey.bmp"
OUT  = "attacked"
os.makedirs(OUT, exist_ok=True)

def save(img, name):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    return path

def log_if_valid(out_path, attack, params, rows):
    p, w = detection(ORIG, WM, out_path)
    valid = int(p == 0 and w >= 35.0)
    print(f"{attack:18} {params:12} -> presence={p}  WPSNR={w:.2f}  VALID={valid}")
    if valid:
        rows.append({
            "Image": "lena_grey.bmp",
            "Group": "shadowmark",
            "Presence": p,
            "WPSNR": round(float(w), 2),
            "Attack": attack,
            "Params": params,
            "OutputFile": out_path,
            "Valid": valid
        })

def main():
    wm_img = cv2.imread(WM, 0)
    if wm_img is None:
        raise FileNotFoundError(f"Missing {WM}. Generate it first with embedding().")

    rows = []

    # singoli (VALID nei tuoi test)
    for qf in [60,50,40,30,20]:
        path = save(A.attack_jpeg(wm_img, qf), f"att_jpeg_qf{qf}.bmp")
        log_if_valid(path, "jpeg", f"qf={qf}", rows)

    path = save(A.attack_resize(wm_img, 0.8), "att_resize_s0.8.bmp")
    log_if_valid(path, "resize", "scale=0.8", rows)

    path = save(A.attack_median(wm_img, 5), "att_median_k5.bmp")
    log_if_valid(path, "median", "ksize=5", rows)

    # combinazioni leggere (VALID nei tuoi test)
    combos = [
        ("resize+jpeg", lambda x: A.attack_jpeg(A.attack_resize(x, 0.6), 60), "scale=0.6,qf=60"),
        ("resize+jpeg", lambda x: A.attack_jpeg(A.attack_resize(x, 0.8), 60), "scale=0.8,qf=60"),
        ("blur+jpeg",   lambda x: A.attack_jpeg(A.attack_blur(x, 3), 70),     "ksize=3,qf=70"),
        ("blur+jpeg",   lambda x: A.attack_jpeg(A.attack_blur(x, 3), 60),     "ksize=3,qf=60"),
    ]
    for name, fn, params in combos:
        path = save(fn(wm_img), f"att_{name}_{params.replace(',','_').replace('=','')}.bmp")
        log_if_valid(path, name, params, rows)

    # strategia (VALID nei tuoi test)
    path = save(A.attack_strategy_3_smart(wm_img), "att_strategy_3_smart.bmp")
    log_if_valid(path, "strategy_3_smart", "", rows)

    # salva CSV solo dei validi
    csv_path = "attack_log_lena_valid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image","Group","Presence","WPSNR","Attack","Params","OutputFile","Valid"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved valid log: {csv_path}  (rows={len(rows)})")

if __name__ == "__main__":
    main()

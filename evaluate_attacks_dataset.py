# evaluate_attacks_dataset.py
import glob, os, csv, cv2, numpy as np
from embedding import embedding
from detection_shadowmark import detection
import attacks as A

IMGS = sorted(glob.glob("images/*.bmp"))[:20] #20 images
#IMGS = sorted(glob.glob("images/*.bmp")) #all images
assert IMGS, "No images found in images/*.bmp"
OUTDIR = "attacked_batch"
os.makedirs(OUTDIR, exist_ok=True)

# definisci qui gli attacchi che vuoi testare in batch
CANDIDATES = [
    ("jpeg60", lambda x: A.attack_jpeg(x, 60)),
    ("jpeg40", lambda x: A.attack_jpeg(x, 40)),
    ("resize08+j60+med3", lambda x: A.attack_resize_jpeg_median(x, 0.6, 60, 3)),
    ("blur3+j60+med3", lambda x: A.attack_blur_jpeg_median(x, 3, 60, 3)),
    ("strategy3", lambda x: A.attack_strategy_3_smart(x)),
]


N_KEYS = 3  # add more for better statistics
rng = np.random.default_rng(2025)

def run():
    summary = {name: {"succ":0, "tot":0, "wps_sum":0.0, "wps_min":999.0} for name,_ in CANDIDATES}
    with open("attack_eval_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["img","attack","presence","wpsnr","valid","outfile"])
        for k in range(N_KEYS):
            key = rng.integers(0, 2, size=1024, dtype=np.uint8)
            np.save("tmp_key.npy", key)
            for p in IMGS:
                wm = embedding(p, "tmp_key.npy")
                cv2.imwrite("tmp_wm.bmp", wm)
                for name, fn in CANDIDATES:
                    try:
                        att = fn(wm)
                    except Exception as e:
                        print(f"[WARN] attack {name} failed on {p}: {e}")
                        continue
                    outp = os.path.join(OUTDIR, f"{os.path.basename(p)[:-4]}__{name}__k{k}.bmp")
                    cv2.imwrite(outp, att)
                    pres, wps = detection(p, "tmp_wm.bmp", outp)
                    valid = int(pres == 0 and wps >= 35.0)
                    wr.writerow([os.path.basename(p), name, pres, round(float(wps), 2), valid, outp])
                    summary[name]["tot"] += 1
                    if valid:
                        summary[name]["succ"] += 1
                        summary[name]["wps_sum"] += float(wps)
                        summary[name]["wps_min"]  = min(summary[name]["wps_min"], float(wps))

    print("\n== SUMMARY ==")
    for name,stat in summary.items():
        tot = stat["tot"]; succ = stat["succ"]
        rate = succ/tot if tot else 0.0
        avgw = (stat["wps_sum"]/succ) if succ else 0.0
        minw = (stat["wps_min"] if succ else 0.0)
        print(f"{name:16}  success={succ:4d}/{tot:4d}  rate={rate:6.2%}  avgWPSNR={avgw:5.2f}  minWPSNR={minw:5.2f}")

if __name__ == "__main__":
    run()

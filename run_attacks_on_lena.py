import os, csv, itertools
import cv2
from detection_shadowmark import detection, wpsnr
import attacks as A

OUTDIR = "attacked"
os.makedirs(OUTDIR, exist_ok=True)

# List of single attacks
single_attacks = [
    ("jpeg", lambda img, p: A.attack_jpeg(img, qf=p["qf"]), {"qf":90}),
    ("jpeg", lambda img, p: A.attack_jpeg(img, qf=p["qf"]), {"qf":80}),
    ("jpeg", lambda img, p: A.attack_jpeg(img, qf=p["qf"]), {"qf":70}),
    ("jpeg", lambda img, p: A.attack_jpeg(img, qf=p["qf"]), {"qf":60}),
    ("awgn", lambda img, p: A.attack_awgn(img, sigma=p["sigma"]), {"sigma":6}),
    ("awgn", lambda img, p: A.attack_awgn(img, sigma=p["sigma"]), {"sigma":10}),
    ("awgn", lambda img, p: A.attack_awgn(img, sigma=p["sigma"]), {"sigma":12}),
    ("awgn", lambda img, p: A.attack_awgn(img, sigma=p["sigma"]), {"sigma":14}),
    ("blur", lambda img, p: A.attack_blur(img, ksize=p["ksize"]), {"ksize":3}),
    ("blur", lambda img, p: A.attack_blur(img, ksize=p["ksize"]), {"ksize":5}),
    ("median", lambda img, p: A.attack_median(img, ksize=p["ksize"]), {"ksize":3}),
    ("median", lambda img, p: A.attack_median(img, ksize=p["ksize"]), {"ksize":5}),
    ("resize", lambda img, p: A.attack_resize(img, scale=p["scale"]), {"scale":0.9}),
    ("resize", lambda img, p: A.attack_resize(img, scale=p["scale"]), {"scale":0.8}),
    ("resize", lambda img, p: A.attack_resize(img, scale=p["scale"]), {"scale":0.7}),
    ("resize", lambda img, p: A.attack_resize(img, scale=p["scale"]), {"scale":0.6}),
    ("resize", lambda img, p: A.attack_resize(img, scale=p["scale"]), {"scale":0.55}),
    ("sharp", lambda img, p: A.attack_sharp(img, amount=p["amount"], radius=p.get("radius",1)), {"amount":1.0, "radius":1}),
    ("sharp", lambda img, p: A.attack_sharp(img, amount=p["amount"], radius=p.get("radius",1)), {"amount":0.6, "radius":1}),

    # strategy examples
    ("strategy_1_stealth", lambda img, p: A.attack_strategy_1_stealth(img), {}),
    ("strategy_3_smart", lambda img, p: A.attack_strategy_3_smart(img), {}),
    ("strategy_4_chaos", lambda img, p: A.attack_strategy_4_chaos(img), {}),
]

# Combined attacks
combined_params = [
    (("resize", {"scale":0.6}), ("jpeg", {"qf":70})),
    (("resize", {"scale":0.6}), ("jpeg", {"qf":80})),
    (("resize", {"scale":0.7}), ("jpeg", {"qf":70})),
]

orig_path = "lena_grey.bmp"
wm_path = "shadowmark_lena_grey.bmp"
orig = cv2.imread(orig_path, 0)
wm = cv2.imread(wm_path, 0)
if orig is None or wm is None:
    raise FileNotFoundError("Make sure lena_grey.bmp and shadowmark_lena_grey.bmp exist")

rows = []
def save_and_log(img, label):
    out_path = os.path.join(OUTDIR, label)
    cv2.imwrite(out_path, img)
    p, w = detection(orig_path, wm_path, out_path)
    valid = 1 if (p==0 and w >= 35.0) else 0
    rows.append({
        "Image":"lena_grey.bmp",
        "Group":"shadowmark",
        "Presence":p,
        "WPSNR":round(float(w),2),
        "Attack": label.split("_")[1] if "_" in label else label,
        "Params": label.split("_",2)[2] if "_" in label else "",
        "OutputFile": out_path,
        "Valid": valid
    })
    print(f"{label} -> presence={p}, WPSNR={w:.2f}, Valid={valid}")

# Run single attacks
for name, func, params in single_attacks:
    label = f"att_{name}"
    if params:
        param_str = "_".join(f"{k}{v}" for k,v in params.items())
        label = f"{label}_{param_str}"
    attacked = func(wm, params)
    save_and_log(attacked, label + ".bmp")

# Run combined attacks
for (a1, p1), (a2, p2) in combined_params:
    def apply_attack_by_name(img, name, params):
        if name == "jpeg": return A.attack_jpeg(img, qf=params["qf"])
        if name == "awgn": return A.attack_awgn(img, sigma=params["sigma"])
        if name == "blur": return A.attack_blur(img, ksize=params["ksize"])
        if name == "median": return A.attack_median(img, ksize=params["ksize"])
        if name == "resize": return A.attack_resize(img, scale=params["scale"])
        return img

    img1 = apply_attack_by_name(wm, a1, p1)
    img2 = apply_attack_by_name(img1, a2, p2)
    label = f"att_{a1}_{list(p1.values())[0]}__{a2}_{list(p2.values())[0]}.bmp"
    save_and_log(img2, label)

csv_path = "attack_log_lena.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Image","Group","Presence","WPSNR","Attack","Params","OutputFile","Valid"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Saved log: {csv_path}")

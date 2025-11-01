import numpy as np
import cv2

# ----------------------
# Helpers
# ----------------------
def _ensure_odd(k):
    k = int(k)
    return k if (k % 2) else (k + 1)

def _clip_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

# ----------------------
# Basic (single) attack functions
# ----------------------
def attack_jpeg(img, qf=20):
    """JPEG compression attack with safe qf clipping."""
    qf = int(np.clip(qf, 5, 95))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    if not ok:
        return img.copy()
    out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out if out is not None else img.copy()

_GLOBAL_RNG = np.random.default_rng(1234)

def attack_awgn(img, sigma=25, rng=None):
    rng = _GLOBAL_RNG if rng is None else rng
    g = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + g
    return _clip_uint8(out)

def attack_blur(img, ksize=13):
    k = _ensure_odd(ksize)
    return cv2.GaussianBlur(img, (k, k), 0)

def attack_median(img, ksize=11):
    k = _ensure_odd(ksize)
    return cv2.medianBlur(img, k)

def attack_resize(img, scale=0.25):
    h, w = img.shape
    nh, nw = max(1, int(h * float(scale))), max(1, int(w * float(scale)))
    # downscale con INTER_AREA
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # upscale con INTER_CUBIC
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def attack_sharp(img, amount=1.0, radius=1):
    blur = cv2.GaussianBlur(img, (0, 0), max(0.1, float(radius)))
    out = cv2.addWeighted(img.astype(np.float32), 1.0 + float(amount),
                          blur.astype(np.float32), -float(amount), 0.0)
    return _clip_uint8(out)

# ----------------------
# Composite / Multi-step attack strategies
# ----------------------
def attack_strategy_1_stealth(img, rng=None):
    rng = _GLOBAL_RNG if rng is None else rng
    img = cv2.GaussianBlur(img, (3, 3), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = rng.normal(0, 5, img.shape).astype(np.float32)
    img = _clip_uint8(img.astype(np.float32) + noise)
    img = cv2.medianBlur(img, 3)
    return img

def attack_strategy_2_brutal(img):
    img = cv2.GaussianBlur(img, (15, 15), 0)
    h, w = img.shape
    small = cv2.resize(img, (max(1,int(w*0.3)), max(1,int(h*0.3))), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = np.random.default_rng().normal(0, 20, img.shape).astype(np.float32)
    img = _clip_uint8(img.astype(np.float32) + noise)
    return img

def attack_strategy_3_smart(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    ok, buf = cv2.imencode(".jpg", img, enc)
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    return img

def attack_strategy_4_chaos(img, rng=None):
    rng = _GLOBAL_RNG if rng is None else rng
    attack_combinations = [
        [lambda x: cv2.GaussianBlur(x, (3,3), 0),
         lambda x: attack_awgn(x, 8, rng=rng),
         lambda x: attack_jpeg(x, 60)],
        [lambda x: cv2.medianBlur(x, 5),
         lambda x: attack_resize(x, 0.7),
         lambda x: attack_awgn(x, 12, rng=rng)],
        [lambda x: cv2.GaussianBlur(x, (7,7), 0),
         lambda x: attack_jpeg(x, 40),
         lambda x: cv2.medianBlur(x, 3)],
        [lambda x: cv2.GaussianBlur(x, (9,9), 0),
         lambda x: attack_awgn(x, 6, rng=rng),
         lambda x: attack_resize(x, 0.5)]
    ]
    idx = int(rng.integers(0, len(attack_combinations)))
    out = img
    for fn in attack_combinations[idx]:
        out = fn(out)
    return out

def attack_strategy_5_precision(img, rng=None):
    rng = _GLOBAL_RNG if rng is None else rng
    img = cv2.GaussianBlur(img, (11, 11), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = rng.normal(0, 15, img.shape).astype(np.float32)
    return _clip_uint8(img.astype(np.float32) + noise)

def attack_strategy_6_wave(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = attack_awgn(img, 5)
    img = cv2.medianBlur(img, 5)
    img = attack_jpeg(img, 50)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = attack_resize(img, 0.5)
    return img

def attack_strategy_7_counter_intuitive(img):
    img = attack_jpeg(img, 80)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.9), int(h*0.9)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    return img

def attack_strategy_8_frequency_hunter(img):
    for k in [3,5,7]:
        img = cv2.GaussianBlur(img, (k,k), 0)
    noise = np.random.default_rng().normal(0, 6, img.shape).astype(np.float32)
    img = _clip_uint8(img.astype(np.float32) + noise)
    img = cv2.medianBlur(img, 3)
    return img

def attack_strategy_9_surgical(img, rng=None):
    rng = _GLOBAL_RNG if rng is None else rng
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = attack_jpeg(img, 50)
    noise = rng.normal(0, 4, img.shape).astype(np.float32)
    return _clip_uint8(img.astype(np.float32) + noise)

# ----------------------
# Useful combo helpers (new)
# ----------------------
def attack_resize_jpeg_median(img, scale=0.6, qf=60, ksize=3):
    out = attack_resize(img, scale)
    out = attack_jpeg(out, qf)
    out = attack_median(out, ksize)
    return out

def attack_blur_jpeg_median(img, k_blur=7, qf=40, k_med=5):
    out = cv2.GaussianBlur(img, (_ensure_odd(k_blur), _ensure_odd(k_blur)), 0)
    out = attack_jpeg(out, qf)
    out = attack_median(out, k_med)
    return out

# ----------------------
# Unified dispatcher
# ----------------------
def attacks(input1, attack_name, param_array=None):
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")

    if attack_name == "strategy_1_stealth":
        return attack_strategy_1_stealth(img)
    if attack_name == "strategy_2_brutal":
        return attack_strategy_2_brutal(img)
    if attack_name == "strategy_3_smart":
        return attack_strategy_3_smart(img)
    if attack_name == "strategy_4_chaos":
        return attack_strategy_4_chaos(img, rng=np.random.default_rng(123))
    if attack_name == "strategy_5_precision":
        return attack_strategy_5_precision(img)
    if attack_name == "strategy_6_wave":
        return attack_strategy_6_wave(img)
    if attack_name == "strategy_7_counter_intuitive":
        return attack_strategy_7_counter_intuitive(img)
    if attack_name == "strategy_8_frequency_hunter":
        return attack_strategy_8_frequency_hunter(img)
    if attack_name == "strategy_9_surgical":
        return attack_strategy_9_surgical(img)

    if attack_name == "sharp":
        amount = param_array[0] if param_array and len(param_array) > 0 else 1.0
        radius = param_array[1] if param_array and len(param_array) > 1 else 1
        return attack_sharp(img, amount=amount, radius=radius)
    if attack_name == "jpeg":
        qf = param_array[0] if param_array else 20
        return attack_jpeg(img, qf)
    if attack_name == "awgn":
        sigma = param_array[0] if param_array else 25
        return attack_awgn(img, sigma)
    if attack_name == "blur":
        ksize = param_array[0] if param_array else 13
        return attack_blur(img, ksize)
    if attack_name == "median":
        ksize = param_array[0] if param_array else 11
        return attack_median(img, ksize)
    if attack_name == "resize":
        scale = param_array[0] if param_array else 0.25
        return attack_resize(img, scale)

    # fallback
    return attack_strategy_1_stealth(img)

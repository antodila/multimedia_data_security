import numpy as np
import cv2

def attack_jpeg(img, qf=20):  # MAXIMUM DESTRUCTION
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)]
    ok, buf = cv2.imencode(".jpg", img, enc)
    if not ok: return img
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def attack_awgn(img, sigma=25):  # MAXIMUM NOISE
    g = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + g
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_blur(img, ksize=13):  # MAXIMUM BLUR
    k = ksize if ksize % 2 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def attack_median(img, ksize=11):  # MAXIMUM MEDIAN
    k = ksize if ksize % 2 else ksize + 1
    return cv2.medianBlur(img, k)

def attack_resize(img, scale=0.25):  # MAXIMUM RESIZE
    h, w = img.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def attack_strategy_1_stealth(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    
    return img

def attack_strategy_2_brutal(img):
    img = cv2.GaussianBlur(img, (15, 15), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.3), int(h*0.3)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def attack_strategy_3_smart(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    ok, buf = cv2.imencode(".jpg", img, enc)
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)    
    return img

def attack_strategy_4_chaos(img):
    attack_combinations = [
        [lambda x: cv2.GaussianBlur(x, (3, 3), 0), lambda x: attack_awgn(x, 8), lambda x: attack_jpeg(x, 60)],
        [lambda x: cv2.medianBlur(x, 5), lambda x: attack_resize(x, 0.7), lambda x: attack_awgn(x, 12)],
        [lambda x: cv2.GaussianBlur(x, (7, 7), 0), lambda x: attack_jpeg(x, 40), lambda x: cv2.medianBlur(x, 3)],
        [lambda x: cv2.GaussianBlur(x, (9, 9), 0), lambda x: attack_awgn(x, 6), lambda x: attack_resize(x, 0.5)]
    ]
    chosen_combination = np.random.choice(len(attack_combinations))
    attacks = attack_combinations[chosen_combination]
    for attack in attacks:
        img = attack(img)
    return img

def attack_strategy_5_precision(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    h, w = img.shape
    small = cv2.resize(img, (int(w*0.8), int(h*0.8)), cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), cv2.INTER_LINEAR)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

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
    for ksize in [3, 5, 7]:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise = np.random.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    return img

def attack_strategy_9_surgical(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = attack_jpeg(img, 50)
    noise = np.random.normal(0, 4, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def attacks(input1, attack_name, param_array=None):
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input1}")
    if attack_name == "strategy_1_stealth":
        return attack_strategy_1_stealth(img)
    elif attack_name == "strategy_2_brutal":
        return attack_strategy_2_brutal(img)
    elif attack_name == "strategy_3_smart":
        return attack_strategy_3_smart(img)
    elif attack_name == "strategy_4_chaos":
        return attack_strategy_4_chaos(img)
    elif attack_name == "strategy_5_precision":
        return attack_strategy_5_precision(img)
    elif attack_name == "strategy_6_wave":
        return attack_strategy_6_wave(img)
    elif attack_name == "strategy_7_counter_intuitive":
        return attack_strategy_7_counter_intuitive(img)
    elif attack_name == "strategy_8_frequency_hunter":
        return attack_strategy_8_frequency_hunter(img)
    elif attack_name == "strategy_9_surgical":
        return attack_strategy_9_surgical(img)
    
    elif attack_name == "jpeg":
        qf = param_array[0] if param_array else 20
        return attack_jpeg(img, qf)
    elif attack_name == "awgn":
        sigma = param_array[0] if param_array else 25
        return attack_awgn(img, sigma)
    elif attack_name == "blur":
        ksize = param_array[0] if param_array else 13
        return attack_blur(img, ksize)
    elif attack_name == "median":
        ksize = param_array[0] if param_array else 11
        return attack_median(img, ksize)
    elif attack_name == "resize":
        scale = param_array[0] if param_array else 0.25
        return attack_resize(img, scale)
    
    else:
        return attack_strategy_1_stealth(img)

#!/usr/bin/env python3
# evaluate_from_log.py (Dynamic Version)
import os, csv, sys, importlib.util
from collections import defaultdict
import cv2
# We must import our OWN wpsnr function from the file we just created.
# This is so our script can calculate WPSNR *if a victim's detector fails*.
from wpsnr import wpsnr as local_wpsnr 

# --- CONFIGURAZIONE ---
IMAGES_DIR    = "images"
DOWNLOAD_DIR  = "download_test"
OUT_DIR       = "attacked_for_submission"
LOG_IN        = "attack_batch_log.csv"
LOG_OUT       = "attack_batch_checked.csv"

# !!! IMPORTANTE: Metti tutti i file .pyc dei detector
#     dei tuoi nemici in questa cartella.
DETECTORS_DIR = "teams" 
# ---

# Cache per i detector già caricati
detector_cache = {}

def load_detector(victim_name):
    """
    Carica dinamicamente il file .pyc del detector di una vittima.
    """
    if victim_name in detector_cache:
        return detector_cache[victim_name]

    # Cerca sia .pyc che .py (il nostro detector di fallback)
    detector_filename = f"detection_{victim_name}.pyc"
    detector_path = os.path.join(DETECTORS_DIR, detector_filename)

    if not os.path.exists(detector_path):
        # Fallback: prova a cercare un file .py (per il nostro detector)
        detector_filename = f"detection_{victim_name}.py"
        detector_path = os.path.join(DETECTORS_DIR, detector_filename)
        
        if not os.path.exists(detector_path):
            print(f"[WARN] File detector non trovato per {victim_name} in: {detector_path}")
            return None

    try:
        # Aggiungi la directory del detector al path di sistema
        # Questo risolve il bug di ACME (ModuleNotFoundError)
        # se wpsnr.py è nella root.
        if DETECTORS_DIR not in sys.path:
             sys.path.append(DETECTORS_DIR)
        
        # Carica il modulo dal file .pyc
        spec = importlib.util.spec_from_file_location(victim_name, detector_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'detection'):
            print(f"[ERROR] 'detection' function not found in {detector_path}")
            return None

        # Salva nella cache e restituisci la funzione
        detector_cache[victim_name] = module.detection
        return module.detection
        
    except Exception as e:
        print(f"[ERROR] Impossibile caricare il detector per {victim_name}: {e}")
        return None

def path_original(image_basename):
    return os.path.join(IMAGES_DIR, image_basename)

def path_watermarked(victim, image_basename):
    return os.path.join(DOWNLOAD_DIR, f"{victim}_{image_basename}")

def path_attacked(attacker, victim, attacked_out):
    return os.path.join(OUT_DIR, attacked_out)

def run():
    # Assicurati che il file wpsnr.py sia presente per i detector "rotti"
    assert os.path.exists("wpsnr.py"), "File wpsnr.py mancante. Crealo dal tuo detection_shadowmark.py"
    assert os.path.exists(LOG_IN), f"Missing {LOG_IN}"
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = defaultdict(lambda: {"tot":0, "succ":0, "wps_sum":0.0, "wps_min":999.0})

    with open(LOG_IN, newline="") as fin, open(LOG_OUT, "w", newline="") as fout:
        rd = csv.DictReader(fin)
        wr = csv.writer(fout)
        wr.writerow(["victim","image","attack_name","params","presence","wpsnr","valid","attacked_path"])

        for row in rd:
            attacker   = row["attacker"]
            victim     = row["victim"]
            orig_wm    = row["original_wm"]
            attacked   = row["attacked_out"]
            attackname = row["attack_name"]
            params     = row.get("params","")

            try:
                image_basename = orig_wm.split("_", 1)[1]
            except Exception:
                print(f"[WARN] nome inatteso original_wm={orig_wm}; salto")
                continue

            p_orig = path_original(image_basename)
            p_wm   = path_watermarked(victim, image_basename)
            p_att  = path_attacked(attacker, victim, attacked)

            missing = [p for p in [p_orig, p_wm, p_att] if not os.path.exists(p)]
            if missing:
                print(f"[WARN] file mancanti per {attacked}: {missing}")
                continue

            # --- Logica di detection dinamica ---
            detector_func = load_detector(victim)
            if detector_func is None:
                # Se non possiamo caricare il detector della vittima, non possiamo verificare.
                continue 

            try:
                # Esegui il detector della VITTIMA
                presence, wps = detector_func(p_orig, p_wm, p_att)
                wps = float(wps)
            except Exception as e:
                print(f"[ERROR] Il detector di {victim} è crashato su {p_att}: {e}")
                # Il detector è rotto. Calcoliamo noi il WPSNR per il log.
                try:
                    wm_img = cv2.imread(p_wm, 0)
                    att_img = cv2.imread(p_att, 0)
                    wps = float(local_wpsnr(wm_img, att_img))
                except:
                    wps = -1.0 # Errore catastrofico
                presence = -1 # Flag per "detector rotto"
            # --- Fine logica dinamica ---

            valid = int(presence == 0 and wps >= 35.0)

            wr.writerow([victim, image_basename, attackname, params, presence, f"{wps:.2f}", valid, p_att])

            s = stats[attackname]
            s["tot"] += 1
            if valid:
                s["succ"] += 1
                s["wps_sum"] += wps
                if wps < s["wps_min"]:
                    s["wps_min"] = wps

    print("\n== SUMMARY by attack_name ==")
    for name, s in sorted(stats.items()):
        tot = s["tot"]; succ = s["succ"]
        rate = (succ / tot) if tot else 0.0
        avgw = (s["wps_sum"]/succ) if succ else 0.0
        minw = (s["wps_min"] if succ else 0.0)
        print(f"{name:20} success={succ:4d}/{tot:4d}  rate={rate:6.2%}  avgWPSNR={avgw:6.2f}  minWPSNR={minw:6.2f}")
    print(f"\nRisultati riga-per-riga in: {LOG_OUT}")

if __name__ == "__main__":
    # Aggiungi la cartella corrente al path
    # Questo aiuta i moduli importati a trovare 'wpsnr.py'
    sys.path.append(os.getcwd()) 
    run()
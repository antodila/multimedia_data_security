import os
import cv2
import glob
import numpy as np
from embedding import embedding 
from detection_shadowmark import detection

# --- Configurazione ---
N_RANDOM_KEYS = 5  # Come richiesto dal prof ("almeno 5 random watermarks")
IMG_FOLDER = "images" # Assicurati che le tue 100+ immagini siano qui
# ---

test_images = glob.glob(os.path.join(IMG_FOLDER, '*.bmp'))
if not test_images:
    print(f"ERRORE: Nessuna immagine .bmp trovata in '{IMG_FOLDER}'")
    exit()

print(f"Inizio verifica FPR su {len(test_images)} immagini e {N_RANDOM_KEYS} watermarks...")

total_count = 0
problem_count = 0

# Loop su 5 diversi watermark casuali
for i in range(N_RANDOM_KEYS):
    print(f"\n--- Test con Watermark Casuale #{i+1} ---")
    
    # 1. Genera un watermark casuale
    watermark = np.random.randint(0, 2, 1024) # 1024 è MARK_SIZE
    np.save('tmp_watermark.npy', watermark)
    WATERMARK_KEY_PATH = 'tmp_watermark.npy'

    img_counter = 0
    
    # Loop su tutte le immagini nel dataset
    for img_path in test_images:
        # Questo è il test del professore:
        # detection(originale, watermarked_casuale, originale)
        
        try:
            # 2. Crea l'immagine con il watermark (necessaria per input2)
            watermarked_image = embedding(img_path, WATERMARK_KEY_PATH)
            watermarked_path = 'temp_watermarked.bmp'
            cv2.imwrite(watermarked_path, watermarked_image)

            # 3. Esegui la detection sull'immagine ORIGINALE
            #    input1 = originale
            #    input2 = watermarked (con chiave casuale)
            #    input3 = originale
            detected_orig, wpsnr_orig = detection(img_path, watermarked_path, img_path)

            total_count += 1
            if detected_orig:
                problem_count += 1
                print(f"  -> (ERRORE FP) Watermark trovato in {os.path.basename(img_path)}")

            img_counter += 1
            if img_counter % 20 == 0:
                print(f"  ...processate {img_counter}/{len(test_images)} immagini")

        except Exception as e:
            print(f"Errore processando {img_path}: {e}")

# --- Risultato Finale ---
print("\n--- Risultato Verifica ---")
if total_count == 0:
    print("ERRORE: Nessun test eseguito.")
else:
    fpr_percent = (problem_count / total_count) * 100
    print(f"Conteggio Falsi Positivi: {problem_count} / {total_count}")
    print(f"FPR Risultante: {fpr_percent:.2f}%")
    
    if fpr_percent < 5.0:
        print("\n✅ CONGRATULAZIONI! L'FPR è sotto il 5%. Problema risolto.")
    else:
        print("\n❌ ATTENZIONE! L'FPR è ancora sopra il 5%.")

# Pulisci i file temporanei
if os.path.exists('tmp_watermark.npy'):
    os.remove('tmp_watermark.npy')
if os.path.exists('temp_watermarked.bmp'):
    os.remove('temp_watermarked.bmp')
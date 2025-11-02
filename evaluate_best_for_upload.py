#!/usr/bin/env python3
"""
evaluate_best_for_upload.py

Legge il file 'attack_batch_checked.csv' (prodotto da evaluate_from_log.py),
identifica l'attacco *valido* (Presenza=0, WPSNR>=35) con il WPSNR
più alto per ogni immagine vittima, e copia quel file in una cartella
'to_upload/' per un facile caricamento.

Produce anche 'upload_list.csv' con un riepilogo dei file selezionati.
"""

import os
import csv
import shutil
from collections import defaultdict

# --- Configurazione ---
LOG_CHECKED = "attack_batch_checked.csv" # Input
LOG_UPLOAD  = "upload_list.csv"          # Output (riepilogo)
SRC_DIR     = "attacked_for_submission"  # Da dove copiare
OUT_DIR     = "to_upload"                # Dove copiare
# ---

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Questo dizionario terrà traccia del miglior attacco per ogni immagine
    # Chiave: (victim, image_basename)
    # Valore: (best_wpsnr, riga_del_csv)
    best_attacks = {}

    print(f"Leggo i risultati da: {LOG_CHECKED}")
    try:
        with open(LOG_CHECKED, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    is_valid = int(row['valid'])
                    wpsnr = float(row['wpsnr'])

                    # Ci interessa solo se l'attacco è valido
                    if is_valid == 1:
                        key = (row['victim'], row['image'])
                        
                        # Se è il primo attacco valido per questa immagine, o se è migliore del precedente
                        if key not in best_attacks or wpsnr > best_attacks[key][0]:
                            best_attacks[key] = (wpsnr, row)
                
                except (ValueError, TypeError):
                    print(f"[WARN] Salto riga malformata: {row}")
                    continue
                    
    except FileNotFoundError:
        print(f"[ERRORE] File non trovato: {LOG_CHECKED}. Esegui prima evaluate_from_log.py")
        return

    if not best_attacks:
        print("Nessun attacco valido (Presence=0, WPSNR>=35) trovato nel log.")
        return

    print(f"Trovati {len(best_attacks)} attacchi validi migliori da copiare.")

    # Scrivi il log di upload e copia i file
    copied_count = 0
    with open(LOG_UPLOAD, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["victim", "image", "best_attacked_filename", "wpsnr", "attack_params"])

        for key, (wpsnr, row) in best_attacks.items():
            victim, image = key
            
            # Estrai il solo nome del file dal percorso completo
            attacked_filename = os.path.basename(row['attacked_path'])
            params = row['params']
            
            # Scrivi la riga nel nuovo CSV
            writer.writerow([victim, image, attacked_filename, f"{wpsnr:.2f}", params])

            # Copia il file
            src_path = os.path.join(SRC_DIR, attacked_filename)
            dst_path = os.path.join(OUT_DIR, attacked_filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"[WARN] File non trovato, impossibile copiare: {src_path}")

    print(f"\nFatto. Creato '{LOG_UPLOAD}' e copiati {copied_count} file in '{OUT_DIR}'.")

if __name__ == "__main__":
    run()
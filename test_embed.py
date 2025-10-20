import cv2
import numpy as np
from embed_spread_spectrum import embedding

# Usa il file lena_grey.bmp come immagine di input
# e il file mark.npy come watermark (o creane uno dummy)

wm = embedding("lena_grey.bmp", "mark.npy")  # genera l’immagine con il watermark

# Salva l’immagine risultante seguendo il nome richiesto dal regolamento
cv2.imwrite("groupA_lena_grey.bmp", wm)

print("Watermarked image saved as groupA_lena_grey.bmp")

import cv2
import attacks as A
from detection_group import detection

# carica la tua immagine watermarked
wm = cv2.imread("groupA_lena_grey.bmp", 0)

# applica l'attacco (puoi provarne altri)
att = A.attack_resize(wm, scale=0.6)
cv2.imwrite("groupA_lena_grey_attacked.bmp", att)

# detection: deve dare presence=0 e WPSNR >= 35
p, w = detection("lena_grey.bmp", "groupA_lena_grey.bmp", "groupA_lena_grey_attacked.bmp")
print("presence:", p, "wpsnr:", w)

from detection_group import detection
p, w = detection("lena_grey.bmp", "groupA_lena_grey.bmp", "groupA_lena_grey.bmp")
print("presence:", p, "wpsnr:", w)

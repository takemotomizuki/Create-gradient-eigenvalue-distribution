from pathlib import Path
import cv2
import numpy as np
from PIL import Image

img_list = list(Path('../ILSVRC2012_img_val').glob('*.JPEG'))
img_list.sort()

for path in img_list:
    img = Image.open(path)
    if img.mode == 'L':
        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
       
    img.save('../ILSVRC2012_img_val_color/' + path.name, quality=95)
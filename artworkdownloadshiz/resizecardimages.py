import PIL
import os
import os.path
from PIL import Image

f = r'C:\Users\mail\OneDrive\Dokumenter\scryfallimgdownload'
for file in os.listdir(f):
    if file.endswith('.jpg'):
        f_img = f+"/"+file
        img = Image.open(f_img)
        img = img.resize((280,190))
        img.save(f_img)
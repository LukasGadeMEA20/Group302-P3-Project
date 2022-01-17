import PIL
import os
from PIL import Image

f = r'C:\Users\mail\OneDrive\Dokumenter\GitHub\Group302-P3-Project\card_data_base\artworkDownloader'
for file in os.listdir(f):
    if file.endswith('.jpg'):
        f_img = f+"/"+file
        print(f_img)
        img = Image.open(f_img)
        img = img.resize((280,190))
        img.save(f_img)
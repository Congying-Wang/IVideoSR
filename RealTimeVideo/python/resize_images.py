import glob
import os
from PIL import Image

dataroot = '../data/predict_img/'
output_dataroot = '../data/predict/'

filenames = glob.glob(dataroot + '*.bmp') + glob.glob(dataroot + '*.jpg')
for name in filenames:
    I = Image.open(name)
    I_resize = I.resize((144, 144), Image.ANTIALIAS)
    current_filename = name.split('/')[-1]
    new_filename = current_filename
    I_resize.save(os.path.join(output_dataroot, 'eval_' + current_filename))


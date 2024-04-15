

from tifffile import TiffFile
from pathlib import Path
from os.path import join, isdir, basename
import os
import tifffile
import xmltodict                               
import torch
import time
from collections import defaultdict
import numpy as np
import pickle

path = 'Orig_data/'

files = os.listdir(path)
output_path = 'data_tubulin/' 


os.makedirs(output_path, exist_ok=True)

study_dirs = list(sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]))

print(study_dirs)

study_dirs = [s for s in study_dirs if s.split('_')[1] in ['4', '7', '17', '20', '21', '24', '29', '30' ]]


images = {}
meta = {}

for s in study_dirs:

    files = os.listdir(os.path.join(path, s))
    print(os.path.join(path, s), files)
    for f in files:
        print(f)
        with tifffile.TiffFile(os.path.join(path, s, f)) as tif:
            metadata   = tif.ome_metadata # Get the existing metadata in a DICT
            metadata = xmltodict.parse(metadata)
            im = tif.asarray()

        i = int(f.split('_')[1])
        m = f.split('_')[2]
        m = m.split('.')[0]
        p = f.split('.')[0]

        meta[p] = metadata

        if i not in images:
            images[i] = {'input_images':[], 'output_images':[]}
        if m in ['Tubulin']: #
            images[i]['output_images'].append((p,m))
        elif m not in ['Nucleus', 'Mitochondria', 'Actin']:
            plane = f.split('_')[3].split('.')[0]
            images[i]['input_images'].append((p,m,plane))
        np.savez(os.path.join(output_path, p), im)
print(images)

with open(os.path.join(output_path, 'images.pkl'), 'wb') as of:
    pickle.dump(images, of)

with open(os.path.join(output_path, 'meta.pkl'), 'wb') as of:
    pickle.dump(meta, of)



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
output_path = 'data/'

os.makedirs(output_path, exist_ok=True)

study_dirs = list(sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]))

print(study_dirs)

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

        meta[i] = (metadata,s,f)

        if i not in images:
            images[i] = {'input_images':[], 'output_images':[]}
        if m in ['Nucleus', 'Mitochondria', 'Actin', 'Tubulin']:
            images[i]['output_images'].append((p,m))
        else:
            plane = f.split('_')[3].split('.')[0]
            images[i]['input_images'].append((p,m,plane))
        np.savez(os.path.join(output_path, p), im)

# Make test-train splits at this point, to ensure consistent between training steps

images_by_output = { m:set() for m in ['Nucleus', 'Mitochondria', 'Actin', 'Tubulin'] }
for i in images:
    for p,m in images[i]['output_images']:
	images_by_output[m].add(i)

print(images_by_output)

# Make test-train split (non-stratified)


	




with open(os.path.join(output_path, 'images.pkl'), 'wb') as of:
    pickle.dump(images, of)

with open(os.path.join(output_path, 'meta.pkl'), 'wb') as of:
    pickle.dump(meta, of)

import os

import numpy as np
from osgeo import gdal

dir_path = "/home/zbl/datasets/STFusion/LGC/original_tif/"
target_dir = '/home/zbl/datasets/STFusion/LGC/LGC_NPY/'
files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
list_dirs = []

for path in files:
    for file in os.listdir(path):
        list_dirs.append(os.path.join(path, file))

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


for image_patch in list_dirs:
    print(f"处理:{image_patch}")
    name = image_patch.split('/')[-1].split('.')[0]
    image = gdal.Open(image_patch)
    image_array = image.ReadAsArray()

    K = np.array(image_array)
    name = name + '.npy'
    np.save(os.path.join(target_dir,name), K)



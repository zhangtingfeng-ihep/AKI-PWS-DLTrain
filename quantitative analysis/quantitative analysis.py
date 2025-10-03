import os
import numpy as np
import tifffile as tiff
from skimage.measure import label


folder_path = r'F:\Sample9\PythonProject\pred\MedT\Comb\label\YZ'
output_file = os.path.join(folder_path, 'comb_MedT_YZ.txt')


def count_vessel_pixels(image):
    return np.sum(image == 1)


def count_disconnected_vessels(image):
    labeled_image = label(image == 1)
    return np.max(labeled_image)


results = []

for i in range(0, 213): 
    file_name = f'YZ{i:04d}.tif'  # 格式化为XZ0000、XZ0001等
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        image = tiff.imread(file_path)
        vessel_pixel_count = count_vessel_pixels(image)
        disconnected_vessel_count = count_disconnected_vessels(image)
        results.append((i, vessel_pixel_count, disconnected_vessel_count))
    else:
        print(f"file {file_name} does not exist")

with open(output_file, 'w') as f:
    f.write("slice_number\tcount_vessel_pixels\tcount_disconnected_vessels\n")
    for slice_num, vessel_pixels, disconnected_vessels in results:
        f.write(f"{slice_num}\t{vessel_pixels}\t{disconnected_vessels}\n")

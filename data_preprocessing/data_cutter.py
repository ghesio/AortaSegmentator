import json
import logging
import math
import os
import numpy as np
import imageio

# Set logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("../cutter.log"),
        logging.StreamHandler()
    ]
)

# Initialize globals
min_x = 99999
max_x = -99999
min_y = 99999
max_y = -99999
min_z = 99999
max_z = -99999


def cut(directory):
    for root, dirs, files in os.walk(directory):
        if files:
            for i in range(len(files) - 1):
                current_image_path = root + '\\' + files[i]
                out_image_path = None
                out_dir = None
                if 'roi' in dir:
                    out_dir = root.replace('roi', 'roi_cut')
                else:
                    out_dir = root.replace('scan', 'scan_cut')
                out_image_path = out_dir + '\\' + files[i]
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # read the image into a numpy array
                current_image = np.array(imageio.imread(uri=current_image_path), dtype='uint8')
                cut_image = None
                if 'axial' in root:
                    cut_image = current_image[center_y - int(new_side_y/2):center_y + int(new_side_y/2),
                                center_x - int(new_side_x/2):center_x + int(new_side_x/2)]
                if 'coronal' in root:
                    cut_image = current_image[center_z - int(new_side_z/2):center_z + int(new_side_z/2),
                                center_x - int(new_side_x/2):center_x + int(new_side_x/2)]
                if 'sagittal' in root:
                    cut_image = current_image[center_z - int(new_side_z/2):center_z + int(new_side_z/2),
                                center_y - int(new_side_y/2):center_y + int(new_side_y/2)]
                logging.debug('Saving image to ' + out_image_path)
                imageio.imwrite(out_image_path, cut_image, format='png')


# dry run flag
dry_run = False
# read JSON containing information
with open('../data/info.json') as f:
    patient_map = json.load(f)
# iterate through every patient and get bounding box vertexes
for patient in patient_map:
    if patient_map[patient]['axial']['min_y'] < min_y:
        min_y = patient_map[patient]['axial']['min_y']
    if patient_map[patient]['axial']['max_y'] > max_y:
        max_y = patient_map[patient]['axial']['max_y']
    if patient_map[patient]['axial']['min_x'] < min_x:
        min_x = patient_map[patient]['axial']['min_x']
    if patient_map[patient]['axial']['max_x'] > max_x:
        max_x = patient_map[patient]['axial']['max_x']
    if patient_map[patient]['coronal']['min_z'] < min_z:
        min_z = patient_map[patient]['coronal']['min_z']
    if patient_map[patient]['coronal']['max_z'] > max_z:
        max_z = patient_map[patient]['coronal']['max_z']
    patient_map[patient]['roi_cut_dir'] = patient_map[patient]['roi_dir'].replace('roi', 'roi_cut')
    patient_map[patient]['scan_cut_dir'] = patient_map[patient]['scan_dir'].replace('scan', 'scan_cut')

logging.info("Updating JSON info file")
with open('../data/info.json', 'w') as outfile:
    json.dump(patient_map, outfile, indent=4)

logging.info("Buonding box location (before padding) (x,y,z): (" + str(min_x) + "-" + str(max_x) + ") x (" +
             str(min_y) + "-" + str(max_y) + ") x (" + str(min_z) + "-" + str(max_z) + ")")

# get max side value to pad to 32px multiple
delta_x = max_x - min_x
delta_y = max_y - min_y
delta_z = max_z - min_z
# calculate center
center_x = math.ceil((max_x + min_x) / 2)
center_y = math.ceil((max_y + min_y) / 2)
center_z = math.ceil((max_z + min_z) / 2)
i = 1
new_side_x = None
new_side_y = None
new_side_z = None
while True:
    if delta_x < 32 * i:
        new_side_x = int(32 * i)
        break
    else:
        i = i+1
i = 1
while True:
    if delta_y < 32 * i:
        new_side_y = int(32 * i)
        break
    else:
        i = i+1
i = 1
while True:
    if delta_z < 32 * i:
        new_side_z = int(32 * i)
        break
    else:
        i = i+1
logging.info("Buonding box location (padded) (x,y,z): (" + str(int(center_x - new_side_x/2)) + "-" + str(int(center_x +
                    new_side_x/2)) + ") x (" + str(int(center_y - new_side_y/2)) + "-" + str(int(center_y +
                    new_side_y/2)) + ") x (" + str(int(center_z - new_side_z/2)) + "-" + str(int(center_z +
                    new_side_z/2)) + ")")
if dry_run:
    exit(1)
# iterate through directories
dir_names = []
for root, dirs, files in os.walk('../data/out'):
    if not dirs:
        dir_names += [os.path.abspath(root)]
for dir in dir_names:
    logging.info('Processing ' + dir)
    cut(dir)
exit(0)

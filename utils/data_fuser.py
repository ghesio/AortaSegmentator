import os
import sys
import shutil
import SimpleITK as sitk
import numpy as np
import imageio
import logging
import math


def fuse_slices(root_path, out_file, out_spacing=(1.0, 1.0, 1.0)):
    # the root path must contains axial, coronal, sagittal directories
    sagittal_files = [root_path + 'sagittal\\' + x for x in os.listdir(root_path + 'sagittal')]
    coronal_files = [root_path + 'coronal\\' + x for x in os.listdir(root_path + 'coronal')]
    axial_files = [root_path + 'axial\\' + x for x in os.listdir(root_path + 'axial')]
    # read all files in different arrays
    sagittal_array = [np.fliplr(np.array(imageio.imread(uri=x), dtype='uint8')) for x in sagittal_files]
    coronal_array = [np.array(imageio.imread(uri=x), dtype='uint8') for x in coronal_files]
    axial_array = [np.flipud(np.array(imageio.imread(uri=x), dtype='uint8')) for x in axial_files]
    axial_shape = np.shape(axial_array)
    combined_axial = np.empty(shape=(axial_shape[0], axial_shape[1], axial_shape[2]))
    combined_coronal = np.empty(shape=(axial_shape[0], axial_shape[1], axial_shape[2]))
    combined_sagittal = np.empty(shape=(axial_shape[0], axial_shape[1], axial_shape[2]))
    for i in range(axial_shape[0]):
        combined_axial[i, :, :] = axial_array[i]
    for i in range(axial_shape[1]):
        combined_coronal[:, i, :] = coronal_array[i]
    for i in range(axial_shape[2]):
        combined_sagittal[:, :, i] = sagittal_array[i]
    # check data integrity
    assert np.array_equal(combined_sagittal, combined_axial) is True \
           and np.array_equal(combined_coronal, combined_axial)
    image = sitk.GetImageFromArray(combined_axial)
    image.SetSpacing(out_spacing)
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    mean_array = (combined_coronal + combined_axial + combined_sagittal) / 3.0
    assert np.array_equal(mean_array, combined_axial)
    sitk.WriteImage(image, out_file)


# this is just a test, EXTREMELY slow
def check_slice_integrity(axial_array, coronal_array, sagittal_array):
    lenght_i = np.shape(axial_array)[0]
    lenght_j = np.shape(coronal_array)[1]
    lenght_k = np.shape(sagittal_array)[2]
    for i in range(lenght_i): #Z
        print('i: {}'.format(i))
        for j in range(lenght_j): #Y
            for k in range(lenght_k): #X
                if axial_array[i][j][k] != coronal_array[j][i][k] and axial_array[i][j][k] != sagittal_array[i][k][j]:
                    exit(-1)
    exit(0)

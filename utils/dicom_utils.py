import math
import os
import shutil
import SimpleITK as sitk
# LOGGING
import imageio
import numpy as np
from utils import custom_logger
import logging
from utils.misc import convert_img


def convert_image_to_numpy_array(input_dir, equalization=True, padding=True, roi=False):
    # Instantiate a DICOM reader
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    # Execute the reader
    image = reader.Execute()
    logging.info('Image series read')
    # Log image size and spacing info BEFORE resample
    size = image.GetSize()
    logging.info('Image size - before resample (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    spacing = image.GetSpacing()
    logging.info('Spacing info - before resample (mm): ' + str(spacing[0]) + 'x' + str(spacing[1]) + 'x' +
                 str(spacing[2]))
    # Resample the image to get 1 px = 1 mm
    image = resample_image(image)
    # Log image size and spacing info after resample to check everything went good
    size = image.GetSize()
    logging.info('Image size - after resample (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    spacing = image.GetSpacing()
    logging.info('Spacing info - after resample (mm): ' + str(spacing[0]) + 'x' + str(spacing[1]) + 'x' +
                 str(spacing[2]))
    # Instantiate and execute orientation filter setting to RAI, to guarantee coherence
    #       across all DICOM images generating slices
    # RAI means
    #   X: patient Right to left
    #   Y: patient Anterior to posterior
    #   Z: patient Inferior to superior
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    # Equalization of the image, this may be slow
    if equalization and not roi:
        logging.info('Equalization of the image')
        sitk.AdaptiveHistogramEqualization(image)
    # Convert to numpy array
    # !!CARE!! the indexing is image_array[z,y,x]
    image_array = sitk.GetArrayFromImage(image)
    # set B/W the image in a ROI
    if roi:
        background_color = image_array[0][0][0]
        image_array[image_array == background_color] = 0
        image_array[image_array != 0] = 255
        pass
    image_array = convert_img(image_array)
    if padding:
        # pad all direction to 32 px multiplier
        i = 1
        while True:
            if image_array.shape[0] < 32 * i:
                new_side_z = int(32 * i)
                break
            else:
                i = i + 1
        i = 1
        while True:
            if image_array.shape[1] < 32 * i:
                new_side_y = int(32 * i)
                break
            else:
                i = i + 1
        i = 1
        while True:
            if image_array.shape[2] < 32 * i:
                new_side_x = int(32 * i)
                break
            else:
                i = i + 1
        # center the image by adding black (0) background
        padding_delta_z = (new_side_z - image_array.shape[0]) / 2
        padding_delta_y = (new_side_y - image_array.shape[1]) / 2
        padding_delta_x = (new_side_x - image_array.shape[2]) / 2
        np_pad = []
        if padding_delta_z.is_integer():
            np_pad.append((int(padding_delta_z), int(padding_delta_z)))
        else:
            np_pad.append((int(padding_delta_z + 1/2), int(padding_delta_z - 1/2)))
        if padding_delta_y.is_integer():
            np_pad.append((int(padding_delta_y), int(padding_delta_y)))
        else:
            np_pad.append((int(padding_delta_y + 1 / 2), int(padding_delta_y - 1 / 2)))
        if padding_delta_x.is_integer():
            np_pad.append((int(padding_delta_x), int(padding_delta_x)))
        else:
            np_pad.append((int(padding_delta_x + 1 / 2), int(padding_delta_x - 1 / 2)))
        image_array = np.pad(image_array, np_pad, mode='constant', constant_values=0)
        padded_shape = image_array.shape
        logging.info('Image size - after padding (px): ' + str(padded_shape[0]) + 'x' +
                     str(padded_shape[1]) + 'x' + str(padded_shape[2]))
    return image_array


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
    """
    Resample itk_image to new out_spacing
    :param itk_image: the input image
    :param out_spacing: the desired spacing
    :return: the resampled image
    """
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)


def save_slices(direction, image_array, root_dir):
    save_directory = root_dir + '/' + direction + '/'
    if os.path.exists(save_directory):
        logging.warning('Deleting old directory ' + save_directory)
        shutil.rmtree(save_directory)
    os.makedirs(save_directory)
    # get max and min value for rescaling
    __range = None
    if direction is 'axial':
        __range = image_array.shape[0]
    elif direction is 'coronal':
        __range = image_array.shape[1]
    else:
        __range = image_array.shape[2]
    logging.info('Start saving ' + direction + ' view in ' + save_directory)
    last_progress = None
    for i in range(__range):
        if math.floor(i*100/__range) % 10 == 0 and math.floor(i*100/__range) != \
                last_progress:
            logging.info(str(math.floor(i*100/__range)) + ' %')
            last_progress = math.floor(i*100/__range)
        output_file_name = save_directory + direction + '_' + str(i).zfill(4) + '.png'
        logging.debug('Saving image to ' + output_file_name)
        # TODO test more the image orientation filter to avoid rotating the images
        if direction is 'axial':
            imageio.imwrite(output_file_name, np.flipud(image_array[i, :, :]), format='png')
        elif direction is 'coronal':
            imageio.imwrite(output_file_name, image_array[:, i, :], format='png')
        else:
            imageio.imwrite(output_file_name, np.fliplr(image_array[:, :, i]), format='png')




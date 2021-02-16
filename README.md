# Aorta Segmentator

## USAGE

* clone the repo
* create a new virtual enviroment, e.g. `python3 -m venv`
* activate it `source venv/bin/activate`
* install required pip packages `pip install -r requirements.txt`
* export the current directory in Python path `export PYTHONPATH="."`
* place the dataset in `data/in` with the folder structure:
  ```
  data
  |____in
  |    |____patient_1
  |    |    |____scan
  |    |    |    | <dicom file series representing the CT scan>
  |    |    |____roi
  |    |         | <dicom file series representing the semiautomatic extracted ROI>
  |    |____patient_2          
  |    |    |____scan
  |    |    |    | <dicom file series representing the CT scan>
  |    |    |____roi
  |    |         | <dicom file series representing the semiautomatic extracted ROI>
    ...
  ```
* run from the root directory `python data_preprocessing/data_slicer.py`
    * this will create in the `data/out` folder the slices for each view (*axial*, *coronal*, *sagittal*)
  ```
  data
  |____out
  |    |____patient_1
  |    |    |____scan
  |    |    |    |____ axial
  |    |    |    |     | axial_0000.png
  |    |    |    |     | axial_0001.png
  |    |    |    |     | ...
  |    |    |    |____ coronal
  |    |    |    |     | coronal_0000.png
  |    |    |    |     | coronal_0001.png
  |    |    |    |     | ...
  |    |    |    |____ sagittal
  |    |    |    |     | sagittal_0000.png
  |    |    |    |     | sagittal_0001.png
  |    |    |    |     | ...
  |    |    |____roi
  |    |    |    |____ axial
  |    |    |    |     | axial_0000.png
  |    |    |    |     | axial_0001.png
  |    |    |    |     | ...
  |    |    |    |____ coronal
  |    |    |    |     | coronal_0000.png
  |    |    |    |     | coronal_0001.png
  |    |    |    |     | ...
  |    |    |    |____ sagittal
  |    |    |    |     | sagittal_0000.png
  |    |    |    |     | sagittal_0001.png
  |    |    |    |     | ...
  |    |____patient_2          
  |    |    | <as above>
  ...
  ```
* run from the root directory `python data_preprocessing/data_locator.py` which will save in the `data` directory
a JSON file named `info.json` to be used in next steps.
  * the file contains a list of the following object
  ```
  "patient_id": {
        "roi_dir": "Root directory  containing the ROI slices",
        "axial": {
            "min_slice": min slices index containing info (not all background),
            "max_slice": max slices index containing info (not all background),
        },
        "coronal": { as above },
        "sagittal": { as above },
        "coordinates" : { contains minimum and maximum not blank informative pixel coordinate }
        "scan_dir": "Root directory containing the scan slices",
        "partition": if the patient belongs to train, validation or test set
    }
  ```
* run from the root directory `python data_preprocessing/data_cutter.py` which will generate for each patient a cut version of the scan
  (`data/out/<patient_id>/scan_cut`) and of the ROI (`data/out/<patient_id>/roi_cut`) by using a bounding box.
  
  The above JSON is enriched with `scan_cut_dir` and `roi_cut_dir` keys which will point to the new created directories.

* run from the root directory `python training/network_training.py` (if needed edit the parameters for epochs, backbone and samples for each patient)
    
  This will train three different unets on axial, coronal and saggital direction; the best results will 
  be saved in `checkpoints` directory with syntax `direction_numberOfSamples_backbone-numberOfEpochs-valLoss.hdf5`

## DETAILS

Only DICOM series are supported.

The input ROI must be a DICOM series which has only 2 colors (an export of a tool like itk-snap is fine).

The data slicer uses (by default) an *adaptive equalization histogram filter* and resample the image pixel values in range 
0-255 (to correctly save a grayscale .png file).

It also normalizes the voxel size of all scan to **1mm x 1mm x 1mm**, so the deep neural network can learn the proper 
spatial semantic, and does a black/white binarization on ROI slices.

In data locator is necessary to specify how many patients are in the validation or test set.

The data cutter uses the information from each object in JSON to create a bounding box for which is guaranteed to contains
all ROIs information (by choosing minimum of the minima and maximum of the maxima for each coordinate).

The network training uses data augmentation by default performing horizontal/vertical shift, rotations and zooms.



For more information read docs and comments in code.

## TODOs

* cleanup unused modules in `requirements.txt` file
* singleton/unified logger
* correct package architecture (may not be possible to run this except from command line, i.e. using PyCharm)
* a predictor which takes a TC scan from disk and saves the predicted ROI
* method to analyse the thresholds extracted from trained networks

## LICENSE
Copyright 2021

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**
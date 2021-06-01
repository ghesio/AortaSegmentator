import SimpleITK as sitk
import sys
import os
import statistics

data_in_dir = '../data/in/'

dir_list = [data_in_dir + d + '/scan/' for d in os.listdir(data_in_dir) if os.path.isdir(os.path.join(data_in_dir, d))]
scan_list = []
for dir in dir_list:
    scan_list.append(dir + [f for f in os.listdir(dir) if not f.startswith('.')][0])

male_age = []
female_age = []


for file in scan_list:
    reader = sitk.ImageFileReader()
    dicom_names = reader.SetFileName(file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    gender = str(reader.GetMetaData('0010|0040')).strip().upper()
    age = int(reader.GetMetaData('0008|0022')[0:4]) - int(reader.GetMetaData('0010|0030')[0:4])
    if gender == 'M':
        male_age.append(age)
    else:
        female_age.append(age)

print('Total ' + str(len(male_age)) + ' M ' + str(len(female_age)) + 'F')
print('Average male age ' + str(statistics.mean(male_age)) + ' ± ' + str(statistics.stdev(male_age)))
print('Average female age ' + str(statistics.mean(female_age)) + ' ± ' + str(statistics.stdev(female_age)))
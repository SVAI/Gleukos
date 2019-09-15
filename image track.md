We use the synapseclient to download the whole dataset programatcially. 


```python
import imageio
import os 
```


```python
os.listdir('WBMRI010')
```




    ['SYNAPSE_METADATA_MANIFEST.tsv', 'DICOM']




```python
os.listdir('WBMRI010/DICOM')
```




    ['Series29-.104.0--', 'SYNAPSE_METADATA_MANIFEST.tsv', 'Series30-.226.0--']




```python
os.listdir('WBMRI010/DICOM/Series29-.104.0--')
```




    ['019-23.0.dcm',
     '009-13.0.dcm',
     '012-16.0.dcm',
     '011-15.0.dcm',
     '020-24.0.dcm',
     '013-17.0.dcm',
     '001-02.0.dcm',
     '007-11.0.dcm',
     'SYNAPSE_METADATA_MANIFEST.tsv',
     '017-21.0.dcm',
     '005-09.0.dcm',
     '008-12.0.dcm',
     '016-20.0.dcm',
     '006-10.0.dcm',
     '010-14.0.dcm',
     '002-06.0.dcm',
     '003-07.0.dcm',
     '014-18.0.dcm',
     '015-19.0.dcm',
     '018-22.0.dcm',
     '004-08.0.dcm']




```python
vol = imageio.volread('WBMRI010/DICOM/Series29-.104.0--')
```

    Reading DICOM (examining files): 1/21 files (4.8%21/21 files (100.0%)
      Found 1 correct series.
    Reading DICOM (loading data): 20/20  (100.0%)


We are interested in three things: shape, samplinga and field of view: 

Image shape: number of elemnts along each axis
sampling rate: physical sapace covered by each element
field of view: physical space covered along each axis 


```python
# Image shape (in voxels)
n0, n1, n2 = vol.shape
n0, n1, n2
```




    (20, 1086, 322)




```python
# Sampling rate (in mm)
d0, d1, d2 = vol.meta['sampling']
d0, d1, d2
```




    (0.0, 1.5625, 1.5625)




```python
# Field of view (in mm) 
n0 * d0, n1 * d1, n2 * d2 

```




    (0.0, 1696.875, 503.125)



## Plotting 


```python
!pip install git+git://github.com/perone/medicaltorch.git
```

    Collecting git+git://github.com/perone/medicaltorch.git
      Cloning git://github.com/perone/medicaltorch.git to /tmp/pip-req-build-o83127xc
      Running command git clone -q git://github.com/perone/medicaltorch.git /tmp/pip-req-build-o83127xc
    Requirement already satisfied: nibabel>=2.2.1 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (2.5.0)
    Requirement already satisfied: scipy>=1.0.0 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (1.3.1)
    Requirement already satisfied: numpy>=1.14.1 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (1.16.4)
    Requirement already satisfied: torch>=0.4.0 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (1.2.0)
    Requirement already satisfied: torchvision>=0.2.1 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (0.4.0a0+6b959ee)
    Requirement already satisfied: tqdm>=4.23.0 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (4.34.0)
    Requirement already satisfied: scikit-image==0.15.0 in /opt/anaconda3/lib/python3.7/site-packages (from medicaltorch==0.2) (0.15.0)
    Requirement already satisfied: six>=1.3 in /opt/anaconda3/lib/python3.7/site-packages (from nibabel>=2.2.1->medicaltorch==0.2) (1.12.0)
    Requirement already satisfied: pillow>=4.1.1 in /opt/anaconda3/lib/python3.7/site-packages (from torchvision>=0.2.1->medicaltorch==0.2) (6.1.0)
    Requirement already satisfied: imageio>=2.0.1 in /opt/anaconda3/lib/python3.7/site-packages (from scikit-image==0.15.0->medicaltorch==0.2) (2.5.0)
    Requirement already satisfied: networkx>=2.0 in /opt/anaconda3/lib/python3.7/site-packages (from scikit-image==0.15.0->medicaltorch==0.2) (2.3)
    Requirement already satisfied: PyWavelets>=0.4.0 in /opt/anaconda3/lib/python3.7/site-packages (from scikit-image==0.15.0->medicaltorch==0.2) (1.0.3)
    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /opt/anaconda3/lib/python3.7/site-packages (from scikit-image==0.15.0->medicaltorch==0.2) (3.1.0)
    Requirement already satisfied: decorator>=4.3.0 in /opt/anaconda3/lib/python3.7/site-packages (from networkx>=2.0->scikit-image==0.15.0->medicaltorch==0.2) (4.4.0)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image==0.15.0->medicaltorch==0.2) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/anaconda3/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image==0.15.0->medicaltorch==0.2) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/anaconda3/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image==0.15.0->medicaltorch==0.2) (2.4.2)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/anaconda3/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image==0.15.0->medicaltorch==0.2) (2.8.0)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib!=3.0.0,>=2.0.0->scikit-image==0.15.0->medicaltorch==0.2) (41.0.1)
    Building wheels for collected packages: medicaltorch
      Building wheel for medicaltorch (setup.py) ... [?25ldone
    [?25h  Stored in directory: /tmp/pip-ephem-wheel-cache-4sum5km4/wheels/48/bc/21/2dfb3f1b41ae62f15ee9a8ed4908e047de7dbb96bf663f3375
    Successfully built medicaltorch
    Installing collected packages: medicaltorch
    Successfully installed medicaltorch-0.2



```python
pip install tensorboardX
```

    Requirement already satisfied: tensorboardX in /opt/anaconda3/lib/python3.7/site-packages (1.8)
    Requirement already satisfied: protobuf>=3.2.0 in /opt/anaconda3/lib/python3.7/site-packages (from tensorboardX) (3.9.1)
    Requirement already satisfied: six in /opt/anaconda3/lib/python3.7/site-packages (from tensorboardX) (1.12.0)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.7/site-packages (from tensorboardX) (1.16.4)
    Requirement already satisfied: setuptools in /opt/anaconda3/lib/python3.7/site-packages (from protobuf>=3.2.0->tensorboardX) (41.0.1)
    Note: you may need to restart the kernel to use updated packages.



```python
from collections import defaultdict
import time
import os

import numpy as np

from tqdm import tqdm

from tensorboardX import SummaryWriter

from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
from medicaltorch import filters as mt_filters

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torchvision.utils as vutils

cudnn.benchmark = True
```

## Let us try to convert the DICOM files to the NII files 


```python
os.listdir('WBMRI010/DICOM/Series29-.104.0--')
```




    ['019-23.0.dcm',
     '009-13.0.dcm',
     '012-16.0.dcm',
     '011-15.0.dcm',
     '020-24.0.dcm',
     '013-17.0.dcm',
     '001-02.0.dcm',
     '007-11.0.dcm',
     'SYNAPSE_METADATA_MANIFEST.tsv',
     '017-21.0.dcm',
     '005-09.0.dcm',
     '008-12.0.dcm',
     '016-20.0.dcm',
     '006-10.0.dcm',
     '010-14.0.dcm',
     '002-06.0.dcm',
     '003-07.0.dcm',
     '014-18.0.dcm',
     '015-19.0.dcm',
     '018-22.0.dcm',
     '004-08.0.dcm']




```python
os.getcwd()
```




    '/home/jupyter/tutorials/image'




```python
import os
full_path = os.getcwd()+'WBMRI010/DICOM/Series29-.104.0--'
```


```python
!dcm2niix Path(full_path)
```

    /bin/sh: 1: Syntax error: "(" unexpected



```python
!dcm2niix /home/jupyter/tutorials/image/WBMRI023/DICOM/
```

    Chris Rorden's dcm2niiX version v1.0.20190902  GCC7.3.0 (64-bit Linux)
    Found 40 DICOM file(s)
    Warning: Siemens MoCo? Bogus slice timing (range -1..-1, TR=4190 seconds)
    Convert 20 DICOM as /home/jupyter/tutorials/image/WBMRI023/DICOM/DICOM_SPINE_20070720114638_16 (322x1105x20x1)
    Warning: Siemens MoCo? Bogus slice timing (range -1..-1, TR=454 seconds)
    Convert 20 DICOM as /home/jupyter/tutorials/image/WBMRI023/DICOM/DICOM_SPINE_20070720114638_22 (322x1102x20x1)
    Conversion required 0.081513 seconds (0.081448 for core code).



```python
!ls /home/tutorials/image
```

    ls: cannot access '/home/tutorials/image': No such file or directory


for i in `<list`; do dcm2niix -f $i /home/jupyter/tutorials/image/$i/DICOM/; done
mv */DICOM/*nii nii

Creating a Dataset for 2D segmentation (slice-wise)
Next we will see how we can create a Dataset using the NIFTI images.
medicaltorch library has a built in generic Dataset called MRI2DSegmentationDataset for this purpose. This dataset takes a tuple of input images and thier labels.


```python
img_list = sorted(os.listdir('/home/jupyter/tutorials/image/nii/a'))
label_list = sorted(os.listdir('/home/jupyter/tutorials/segmentation/segmentation_50cases'))
```

We need to check the length so that we have one-to-one match between segmented and WBMRI data 


```python
print(len(img_list))
```

    48



```python
print(len(label_list))
```

    51



```python
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
print(Diff(label_list, img_list))
```

    ['036.nii', '032.nii', '052.nii']



```python
for element in Diff(label_list, img_list): 
    cleaned_label_list=label_list.remove(element)
```


```python
filename_pairs = [(os.path.join('/home/jupyter/tutorials/image/nii/b',x),os.path.join('/home/jupyter/tutorials/segmentation/segmentation_50cases',y)) for x,y in zip(img_list,label_list)]
filename_pairs 
```




    [('/home/jupyter/tutorials/image/nii/b/010.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/010.nii'),
     ('/home/jupyter/tutorials/image/nii/b/019.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/019.nii'),
     ('/home/jupyter/tutorials/image/nii/b/023.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/023.nii'),
     ('/home/jupyter/tutorials/image/nii/b/027.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/027.nii'),
     ('/home/jupyter/tutorials/image/nii/b/035.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/035.nii'),
     ('/home/jupyter/tutorials/image/nii/b/045.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/045.nii'),
     ('/home/jupyter/tutorials/image/nii/b/046.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/046.nii'),
     ('/home/jupyter/tutorials/image/nii/b/047.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/047.nii'),
     ('/home/jupyter/tutorials/image/nii/b/048.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/048.nii'),
     ('/home/jupyter/tutorials/image/nii/b/053.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/053.nii'),
     ('/home/jupyter/tutorials/image/nii/b/061.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/061.nii'),
     ('/home/jupyter/tutorials/image/nii/b/063.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/063.nii'),
     ('/home/jupyter/tutorials/image/nii/b/065.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/065.nii'),
     ('/home/jupyter/tutorials/image/nii/b/069.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/069.nii'),
     ('/home/jupyter/tutorials/image/nii/b/072.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/072.nii'),
     ('/home/jupyter/tutorials/image/nii/b/076.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/076.nii'),
     ('/home/jupyter/tutorials/image/nii/b/077.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/077.nii'),
     ('/home/jupyter/tutorials/image/nii/b/081.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/081.nii'),
     ('/home/jupyter/tutorials/image/nii/b/085.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/085.nii'),
     ('/home/jupyter/tutorials/image/nii/b/087.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/087.nii'),
     ('/home/jupyter/tutorials/image/nii/b/088.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/088.nii'),
     ('/home/jupyter/tutorials/image/nii/b/089.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/089.nii'),
     ('/home/jupyter/tutorials/image/nii/b/090.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/090.nii'),
     ('/home/jupyter/tutorials/image/nii/b/095.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/095.nii'),
     ('/home/jupyter/tutorials/image/nii/b/097.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/097.nii'),
     ('/home/jupyter/tutorials/image/nii/b/098.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/098.nii'),
     ('/home/jupyter/tutorials/image/nii/b/099.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/099.nii'),
     ('/home/jupyter/tutorials/image/nii/b/102.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/102.nii'),
     ('/home/jupyter/tutorials/image/nii/b/106.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/106.nii'),
     ('/home/jupyter/tutorials/image/nii/b/107.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/107.nii'),
     ('/home/jupyter/tutorials/image/nii/b/109.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/109.nii'),
     ('/home/jupyter/tutorials/image/nii/b/114.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/114.nii'),
     ('/home/jupyter/tutorials/image/nii/b/119.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/119.nii'),
     ('/home/jupyter/tutorials/image/nii/b/120.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/120.nii'),
     ('/home/jupyter/tutorials/image/nii/b/123.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/123.nii'),
     ('/home/jupyter/tutorials/image/nii/b/127.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/127.nii'),
     ('/home/jupyter/tutorials/image/nii/b/136.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/136.nii'),
     ('/home/jupyter/tutorials/image/nii/b/138.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/138.nii'),
     ('/home/jupyter/tutorials/image/nii/b/144.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/144.nii'),
     ('/home/jupyter/tutorials/image/nii/b/158.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/158.nii'),
     ('/home/jupyter/tutorials/image/nii/b/196.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/196.nii'),
     ('/home/jupyter/tutorials/image/nii/b/222.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/222.nii'),
     ('/home/jupyter/tutorials/image/nii/b/224.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/224.nii'),
     ('/home/jupyter/tutorials/image/nii/b/225.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/225.nii'),
     ('/home/jupyter/tutorials/image/nii/b/227.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/227.nii'),
     ('/home/jupyter/tutorials/image/nii/b/268.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/268.nii'),
     ('/home/jupyter/tutorials/image/nii/b/279.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/279.nii')]




```python
# transformer
composed_transform = transforms.Compose([
            mt_transforms.Resample(0.25, 0.25),
            mt_transforms.CenterCrop2D((200, 200)),
            mt_transforms.ToTensor(),
])
```


```python
# load data
train_dataset = mt_datasets.MRI3DSegmentationDataset(filename_pairs,transform=composed_transform)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-20-a0e71e91cf42> in <module>
          1 # load data
    ----> 2 train_dataset = mt_datasets.MRI3DSegmentationDataset(filename_pairs,transform=composed_transform)
    

    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in __init__(self, filename_pairs, cache, transform, canonical)
        334         self.canonical = canonical
        335 
    --> 336         self._load_filenames()
        337 
        338     def _load_filenames(self):


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in _load_filenames(self)
        339         for input_filename, gt_filename in self.filename_pairs:
        340             segpair = SegmentationPair2D(input_filename, gt_filename,
    --> 341                                          self.cache, self.canonical)
        342             self.handlers.append(segpair)
        343 


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in __init__(self, input_filename, gt_filename, cache, canonical)
         90         if self.gt_handle is not None:
         91             if not np.allclose(input_shape, gt_shape):
    ---> 92                 raise RuntimeError('Input and ground truth with different dimensions.')
         93 
         94         if self.canonical:


    RuntimeError: Input and ground truth with different dimensions.


We ran into compatiblity issues 


```python
pip install SimpleITK
```

    Collecting SimpleITK
    [?25l  Downloading https://files.pythonhosted.org/packages/06/7f/900f97ec9c88b398b79039834d5aa8b186589d57a7b85f98e02b93bf114c/SimpleITK-1.2.2-cp37-cp37m-manylinux1_x86_64.whl (42.5MB)
    [K     |████████████████████████████████| 42.5MB 4.9MB/s eta 0:00:01
    [?25hInstalling collected packages: SimpleITK
    Successfully installed SimpleITK-1.2.2
    Note: you may need to restart the kernel to use updated packages.


This is for image in the label dataset 


```python
import SimpleITK as sitk
for element in label_list[1:]:
    file=os.path.join('/home/jupyter/tutorials/segmentation/segmentation_50cases/',element)
    nda = sitk.GetArrayFromImage(sitk.ReadImage(file))
    nda = nda[1:20,0:894,1:321]
    img = sitk.GetImageFromArray(nda)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file)
    writer.Execute(img)
```

This is for image in b (origionallly without a) 


```python
import SimpleITK as sitk
for element in img_list[1:]:
    file=os.path.join('/home/jupyter/tutorials/image/nii/b',element)
    nda = sitk.GetArrayFromImage(sitk.ReadImage(file))
    print(sitk.ReadImage(file).GetSize())
    print(nda.shape)
    nda = nda[1:18,0:894,1:322]
    img = sitk.GetImageFromArray(nda)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file)
    writer.Execute(img)
```

    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)
    (320, 894, 19)
    (19, 894, 320)


Just checking if two sets are of the same shape now


```python
import imageio
vol2 = imageio.volread('/home/jupyter/tutorials/segmentation/segmentation_50cases/010.nii')
# Image shape (in voxels)
n0, n1, n2 = vol2.shape
n0, n1, n2
```




    (17, 894, 317)




```python
import imageio
vol2 = imageio.volread('/home/jupyter/tutorials/image/nii/b/010.nii')
# Image shape (in voxels)
n0, n1, n2 = vol2.shape
n0, n1, n2
```




    (17, 894, 317)




```python

```


```python
import SimpleITK as sitk
for element in img_list[1:]:
    file=os.path.join('/home/jupyter/tutorials/image/nii/b',element)
    nda = sitk.GetArrayFromImage(sitk.ReadImage(file))
    nda = nda[:,:,0:317]
    print(nda.shape)
    img = sitk.GetImageFromArray(nda)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file)
    writer.Execute(img)
```

    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)



```python
import SimpleITK as sitk
for element in label_list[1:]:
    file=os.path.join('/home/jupyter/tutorials/segmentation/segmentation_50cases/',element)
    nda = sitk.GetArrayFromImage(sitk.ReadImage(file))
    nda = nda[:,:,0:317]
    print(nda.shape)
    img = sitk.GetImageFromArray(nda)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file)
    writer.Execute(img)
```

    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)
    (17, 894, 317)



```python
print(len(img_list))
```

    48



```python
print(len(label_list))
```

    51



```python
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
print(Diff(label_list, img_list))
```

    ['036.nii', '032.nii', '052.nii']



```python
for element in Diff(label_list, img_list): 
    cleaned_label_list=label_list.remove(element)
```


```python
filename_pairs = [(os.path.join('/home/jupyter/tutorials/image/nii/b',x),os.path.join('/home/jupyter/tutorials/segmentation/segmentation_50cases',y)) for x,y in zip(img_list,label_list)]
filename_pairs = filename_pairs[1:]
filename_pairs 
```




    [('/home/jupyter/tutorials/image/nii/b/010.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/010.nii'),
     ('/home/jupyter/tutorials/image/nii/b/019.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/019.nii'),
     ('/home/jupyter/tutorials/image/nii/b/023.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/023.nii'),
     ('/home/jupyter/tutorials/image/nii/b/027.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/027.nii'),
     ('/home/jupyter/tutorials/image/nii/b/035.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/035.nii'),
     ('/home/jupyter/tutorials/image/nii/b/045.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/045.nii'),
     ('/home/jupyter/tutorials/image/nii/b/046.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/046.nii'),
     ('/home/jupyter/tutorials/image/nii/b/047.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/047.nii'),
     ('/home/jupyter/tutorials/image/nii/b/048.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/048.nii'),
     ('/home/jupyter/tutorials/image/nii/b/053.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/053.nii'),
     ('/home/jupyter/tutorials/image/nii/b/061.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/061.nii'),
     ('/home/jupyter/tutorials/image/nii/b/063.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/063.nii'),
     ('/home/jupyter/tutorials/image/nii/b/065.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/065.nii'),
     ('/home/jupyter/tutorials/image/nii/b/069.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/069.nii'),
     ('/home/jupyter/tutorials/image/nii/b/072.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/072.nii'),
     ('/home/jupyter/tutorials/image/nii/b/076.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/076.nii'),
     ('/home/jupyter/tutorials/image/nii/b/077.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/077.nii'),
     ('/home/jupyter/tutorials/image/nii/b/081.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/081.nii'),
     ('/home/jupyter/tutorials/image/nii/b/085.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/085.nii'),
     ('/home/jupyter/tutorials/image/nii/b/087.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/087.nii'),
     ('/home/jupyter/tutorials/image/nii/b/088.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/088.nii'),
     ('/home/jupyter/tutorials/image/nii/b/089.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/089.nii'),
     ('/home/jupyter/tutorials/image/nii/b/090.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/090.nii'),
     ('/home/jupyter/tutorials/image/nii/b/095.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/095.nii'),
     ('/home/jupyter/tutorials/image/nii/b/097.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/097.nii'),
     ('/home/jupyter/tutorials/image/nii/b/098.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/098.nii'),
     ('/home/jupyter/tutorials/image/nii/b/099.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/099.nii'),
     ('/home/jupyter/tutorials/image/nii/b/102.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/102.nii'),
     ('/home/jupyter/tutorials/image/nii/b/106.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/106.nii'),
     ('/home/jupyter/tutorials/image/nii/b/107.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/107.nii'),
     ('/home/jupyter/tutorials/image/nii/b/109.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/109.nii'),
     ('/home/jupyter/tutorials/image/nii/b/114.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/114.nii'),
     ('/home/jupyter/tutorials/image/nii/b/119.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/119.nii'),
     ('/home/jupyter/tutorials/image/nii/b/120.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/120.nii'),
     ('/home/jupyter/tutorials/image/nii/b/123.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/123.nii'),
     ('/home/jupyter/tutorials/image/nii/b/127.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/127.nii'),
     ('/home/jupyter/tutorials/image/nii/b/136.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/136.nii'),
     ('/home/jupyter/tutorials/image/nii/b/138.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/138.nii'),
     ('/home/jupyter/tutorials/image/nii/b/144.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/144.nii'),
     ('/home/jupyter/tutorials/image/nii/b/158.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/158.nii'),
     ('/home/jupyter/tutorials/image/nii/b/196.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/196.nii'),
     ('/home/jupyter/tutorials/image/nii/b/222.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/222.nii'),
     ('/home/jupyter/tutorials/image/nii/b/224.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/224.nii'),
     ('/home/jupyter/tutorials/image/nii/b/225.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/225.nii'),
     ('/home/jupyter/tutorials/image/nii/b/227.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/227.nii'),
     ('/home/jupyter/tutorials/image/nii/b/268.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/268.nii'),
     ('/home/jupyter/tutorials/image/nii/b/279.nii',
      '/home/jupyter/tutorials/segmentation/segmentation_50cases/279.nii')]




```python
# transformer
composed_transform = transforms.Compose([
            mt_transforms.Resample(0.25, 0.25),
            mt_transforms.CenterCrop2D((200, 200)),
            mt_transforms.ToTensor(),
])
```


```python
# load data
train_dataset = mt_datasets.MRI3DSegmentationDataset(filename_pairs,transform=composed_transform)
```

https://github.com/perone/medicaltorch/blob/master/examples/Dataloaders_NIFTI.ipynb


```python
type(train_dataset)

```




    medicaltorch.datasets.MRI3DSegmentationDataset




```python
print(len(train_dataset))

```

    47



The next step is to create a PyTorch Dataloader over this Dataset and let PyTorch do its magic


```python
dataloader = DataLoader(train_dataset, batch_size=2,collate_fn=mt_datasets.mt_collate)
batch = next(iter(dataloader))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-114-a3932a73b4b5> in <module>
          1 dataloader = DataLoader(train_dataset, batch_size=2,collate_fn=mt_datasets.mt_collate)
    ----> 2 batch = next(iter(dataloader))
    

    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        344     def __next__(self):
        345         index = self._next_index()  # may raise StopIteration
    --> 346         data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        347         if self.pin_memory:
        348             data = _utils.pin_memory.pin_memory(data)


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in __getitem__(self, index)
        365         }
        366         if self.transform is not None:
    --> 367             data_dict = self.transform(data_dict)
        368         return data_dict
        369 


    /opt/anaconda3/lib/python3.7/site-packages/torchvision/transforms/transforms.py in __call__(self, img)
         59     def __call__(self, img):
         60         for t in self.transforms:
    ---> 61             img = t(img)
         62         return img
         63 


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/transforms.py in __call__(self, sample)
        640         rdict = {}
        641         input_data = sample['input']
    --> 642         input_metadata = sample['input_metadata']
        643 
        644         # Voxel dimension in mm


    KeyError: 'input_metadata'



```python
# PyTorch data loader
dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=4,
                        collate_fn=mt_datasets.mt_collate)
minibatch = next(iter(dataloader))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-115-5acdbbe5a5d6> in <module>
          3                         shuffle=True, num_workers=4,
          4                         collate_fn=mt_datasets.mt_collate)
    ----> 5 minibatch = next(iter(dataloader))
    

    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        817             else:
        818                 del self.task_info[idx]
    --> 819                 return self._process_data(data)
        820 
        821     next = __next__  # Python 2 compatibility


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py in _process_data(self, data)
        844         self._try_put_index()
        845         if isinstance(data, ExceptionWrapper):
    --> 846             data.reraise()
        847         return data
        848 


    /opt/anaconda3/lib/python3.7/site-packages/torch/_utils.py in reraise(self)
        367             # (https://bugs.python.org/issue2651), so we work around it.
        368             msg = KeyErrorMessage(msg)
    --> 369         raise self.exc_type(msg)
    

    KeyError: Caught KeyError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py", line 178, in _worker_loop
        data = fetcher.fetch(index)
      File "/opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 44, in fetch
        data = [self.dataset[idx] for idx in possibly_batched_index]
      File "/opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 44, in <listcomp>
        data = [self.dataset[idx] for idx in possibly_batched_index]
      File "/opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py", line 367, in __getitem__
        data_dict = self.transform(data_dict)
      File "/opt/anaconda3/lib/python3.7/site-packages/torchvision/transforms/transforms.py", line 61, in __call__
        img = t(img)
      File "/opt/anaconda3/lib/python3.7/site-packages/medicaltorch/transforms.py", line 642, in __call__
        input_metadata = sample['input_metadata']
    KeyError: 'input_metadata'




```python
help(dataloader)
```

    Help on DataLoader in module torch.utils.data.dataloader object:
    
    class DataLoader(builtins.object)
     |  DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
     |  
     |  Data loader. Combines a dataset and a sampler, and provides an iterable over
     |  the given dataset.
     |  
     |  The :class:`~torch.utils.data.DataLoader` supports both map-style and
     |  iterable-style datasets with single- or multi-process loading, customizing
     |  loading order and optional automatic batching (collation) and memory pinning.
     |  
     |  See :py:mod:`torch.utils.data` documentation page for more details.
     |  
     |  Arguments:
     |      dataset (Dataset): dataset from which to load the data.
     |      batch_size (int, optional): how many samples per batch to load
     |          (default: ``1``).
     |      shuffle (bool, optional): set to ``True`` to have the data reshuffled
     |          at every epoch (default: ``False``).
     |      sampler (Sampler, optional): defines the strategy to draw samples from
     |          the dataset. If specified, :attr:`shuffle` must be ``False``.
     |      batch_sampler (Sampler, optional): like :attr:`sampler`, but returns a batch of
     |          indices at a time. Mutually exclusive with :attr:`batch_size`,
     |          :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
     |      num_workers (int, optional): how many subprocesses to use for data
     |          loading. ``0`` means that the data will be loaded in the main process.
     |          (default: ``0``)
     |      collate_fn (callable, optional): merges a list of samples to form a
     |          mini-batch of Tensor(s).  Used when using batched loading from a
     |          map-style dataset.
     |      pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
     |          into CUDA pinned memory before returning them.  If your data elements
     |          are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
     |          see the example below.
     |      drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
     |          if the dataset size is not divisible by the batch size. If ``False`` and
     |          the size of dataset is not divisible by the batch size, then the last batch
     |          will be smaller. (default: ``False``)
     |      timeout (numeric, optional): if positive, the timeout value for collecting a batch
     |          from workers. Should always be non-negative. (default: ``0``)
     |      worker_init_fn (callable, optional): If not ``None``, this will be called on each
     |          worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
     |          input, after seeding and before data loading. (default: ``None``)
     |  
     |  
     |  .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
     |               cannot be an unpicklable object, e.g., a lambda function. See
     |               :ref:`multiprocessing-best-practices` on more details related
     |               to multiprocessing in PyTorch.
     |  
     |  .. note:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
     |            When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
     |            an infinite sampler is used, whose :meth:`__len__` is not
     |            implemented, because the actual length depends on both the
     |            iterable as well as multi-process loading configurations. So one
     |            should not query this method unless they work with a map-style
     |            dataset. See `Dataset Types`_ for more details on these two types
     |            of datasets.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  __setattr__(self, attr, val)
     |      Implement setattr(self, name, value).
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  multiprocessing_context
    



```python
# PyTorch data loader
dataloader = DataLoader(train_dataset, batch_size=2,collate_fn=mt_datasets.mt_collate)
```


```python
dataloader
```




    <torch.utils.data.dataloader.DataLoader at 0x7f17b6fa33d0>



# Constructing The Segmentation Model

We saw the images above, now we want to build the gray matter segmentation model with the MRI images provided above. Let's define a helper function that helps to decide the final predictions of the model.



```python
def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds
```


```python
train_transform = transforms.Compose([
        mt_transforms.Resample(0.25, 0.25),
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
])

val_transform = transforms.Compose([
        mt_transforms.Resample(0.25, 0.25),
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
])
```

# Model and Parameters
Below we declare our model and parameters. Note that we are using GPU in this notebook. Also note that the model used below refers to the U-net convolutional-based architecture proposed by Ronneberger et al., 2015, which essentially aggregates semantic information to perform the segmentation. See a figure of the U-net framework below. You can also refer to the medicaltorch API documentation for more available state-of-the-art implementations.


```python
model = mt_models.Unet(drop_rate=0.4, bn_momentum=0.1)
model.cuda()
num_epochs = 10
initial_lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
```


```python
def numeric_score(prediction, groundtruth):
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN 
  
def accuracy(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0
```

# Training
Now we finally train the model for spinal cord gray matter segmentation. We report the training and testing accuracy below and train for 10 epochs only.


```python
for epoch in tqdm(range(1, num_epochs+1)):
    start_time = time.time()

    scheduler.step()

    lr = scheduler.get_lr()[0]

    model.train()
    train_loss_total = 0.0
    num_steps = 0
    
    ### Training
    for i, batch in enumerate(dataloader):
        input_samples, gt_samples = batch["input"], batch["gt"]

        var_input = input_samples.cuda()
        var_gt = gt_samples.cuda()

        preds = model(var_input)

        loss = mt_losses.dice_loss(preds, var_gt)
        train_loss_total += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_steps += 1

        if epoch % 5 == 0:
            grid_img = vutils.make_grid(input_samples,
                                        normalize=True,
                                        scale_each=True)
            

            grid_img = vutils.make_grid(preds.data.cpu(),
                                        normalize=True,
                                        scale_each=True)
            

            grid_img = vutils.make_grid(gt_samples,
                                        normalize=True,
                                        scale_each=True)
   
    
    train_loss_total_avg = train_loss_total / num_steps
    model.eval()
    val_loss_total = 0.0
    num_steps = 0
    train_acc  = accuracy(preds.cpu().detach().numpy(), 
                          var_gt.cpu().detach().numpy())
    
    metric_fns = [mt_metrics.dice_score,
                  mt_metrics.hausdorff_score,
                  mt_metrics.precision_score,
                  mt_metrics.recall_score,
                  mt_metrics.specificity_score,
                  mt_metrics.intersection_over_union,
                  mt_metrics.accuracy_score]

    metric_mgr = mt_metrics.MetricManager(metric_fns)
```

      0%|          | 0/10 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-129-3aa05a95fdf0> in <module>
         11 
         12     ### Training
    ---> 13     for i, batch in enumerate(dataloader):
         14         input_samples, gt_samples = batch["input"], batch["gt"]
         15 


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        344     def __next__(self):
        345         index = self._next_index()  # may raise StopIteration
    --> 346         data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        347         if self.pin_memory:
        348             data = _utils.pin_memory.pin_memory(data)


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in __getitem__(self, index)
        365         }
        366         if self.transform is not None:
    --> 367             data_dict = self.transform(data_dict)
        368         return data_dict
        369 


    /opt/anaconda3/lib/python3.7/site-packages/torchvision/transforms/transforms.py in __call__(self, img)
         59     def __call__(self, img):
         60         for t in self.transforms:
    ---> 61             img = t(img)
         62         return img
         63 


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/transforms.py in __call__(self, sample)
        640         rdict = {}
        641         input_data = sample['input']
    --> 642         input_metadata = sample['input_metadata']
        643 
        644         # Voxel dimension in mm


    KeyError: 'input_metadata'



```python
next(iter(dataloader))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-130-388aca337e2f> in <module>
    ----> 1 next(iter(dataloader))
    

    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
        344     def __next__(self):
        345         index = self._next_index()  # may raise StopIteration
    --> 346         data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        347         if self.pin_memory:
        348             data = _utils.pin_memory.pin_memory(data)


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/datasets.py in __getitem__(self, index)
        365         }
        366         if self.transform is not None:
    --> 367             data_dict = self.transform(data_dict)
        368         return data_dict
        369 


    /opt/anaconda3/lib/python3.7/site-packages/torchvision/transforms/transforms.py in __call__(self, img)
         59     def __call__(self, img):
         60         for t in self.transforms:
    ---> 61             img = t(img)
         62         return img
         63 


    /opt/anaconda3/lib/python3.7/site-packages/medicaltorch/transforms.py in __call__(self, sample)
        640         rdict = {}
        641         input_data = sample['input']
    --> 642         input_metadata = sample['input_metadata']
        643 
        644         # Voxel dimension in mm


    KeyError: 'input_metadata'



```python

```

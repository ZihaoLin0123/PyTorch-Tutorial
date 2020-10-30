# -*- coding: utf-8 -*-
"""
# @file name  : 03_transforms_1.py
# @author     : zihao lin (quoted from tingsongyu)
# @date       : Oct. 29th 2020 21:18:00
# @brief      : transforms method 1
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


path_lenet = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "model", "lenet.py"))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "tools", "common_tools.py"))
assert os.path.exists(path_lenet), "{} does not exists, please add lenet.py file to {}".format(path_lenet, os.path.dirname(path_lenet))
assert os.path.exists(path_tools), "{}does not exists, please add common_tools.py file to {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

'''
Part 1: Crop

1. transforms.CenterCrop

    Function: cut the picture from center
    
    Size: the size of the new picture after cut
    
2. transforms.RandomCrop

    transforms.RandomCrop(size,
                          padding=None,
                          pad_if_needed=False,
                          fill=0,
                          padding_mode='constant')
                          
    Function: cut the picture randomly to get a new picture with size
    
    Parameters:
        a. size: the size of the new picture after cut
        b. padding: configure the size of padding
            (i) if padding = a, fill a pixels in up, down, left and right
            (ii) if padding = (a, b), fill b pixels in up and down, fill a pixels in left and right
            (iii) if padding = (a, b, c, d), fill a, b, c and d pixels in left, up, right and down separately
        c. pad_if_need: if picture is smaller than size, then fill

3. RandomResizedCrop

    RandomResizedCrop(size,
                      scale=(0.08, 1.0),
                      ratio=(3/4, 4/3),
                      interpolation)
                      
    Function: cut picture randomly by size and the ratio of length and width
    
    Parameters:
        a. size: the size of the new picture after cut
        b. scale: the ratio of area of cutting randomly, default (0.08, 1)
        c. ratio: the ratio of length over width, default(3/4, 4/3)
        d. interpolation: the way of interpolation
            (i) PIL.Image.NEAREST
            (ii) PIL.Image.BILINEAR
            (iii) PIL.Image.BICUBIC
            
4. FiveCrop

    transforms.FiveCrop(size)

5. TenCrop

    transforms.TenCrop(size,
                       vertical_flip=False)
                       
    Function: FiveCrop cuts the original picture into five pictures (from up, down, left, right & center) of size.
              TenCrop reverses the 5 pictures vertical or horizontal to get 10 pictures.


Part 2: Flip and Rotation

1. RandomHorizontalFlip

    RandomHorizontalFlip(p=0.5)
    
2. RandomVerticalFlip

    RandomVerticalFlip(p=0.5)
    
    Function: flip the picture horizontal & vertical with a probability
    
3. RandomRotation

    RandomRotation(degrees,
                   resample=False,
                   expand=False,
                   center=None)
                   
    Function: rotate the picture randomly
    
    Parameters:
        a. degrees: the degree of rotation
           when degrees = a, rotate degree randomly from (-a, a)
           when degrees = (a, b), rotate degree randomly from (a, b)
        b. resample: resample method
        c. expand: whether to expand the picture to maintain the information of original picture
        d. center: the center of rotation
'''

set_seed(1)  # set random seed

# set parameters
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}

# ============================ step 1/5 data ============================
split_dir = os.path.abspath(os.path.join("..", "..", "data", "RMB_split"))
if not os.path.exists(split_dir):
    raise Exception(r"data {} does not exists, please go back to lesson02\01_RMB_classification_data.py \
                    to generate the data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transformorms.CenterCrop(512),     # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),  # this function returns a tuple data which could not directly work
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
    # torch.stack concatenate the tensors in a dimension

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    # do not need these two lines when run FiveCrop and TenCrop
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# create MyDataset instance
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# create DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 train ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        # do not use the following six lines when run FiveCrop or TenCrop, instead using the next function
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)
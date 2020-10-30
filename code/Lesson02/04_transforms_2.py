# -*- coding: utf-8 -*-
"""
# @file name  : 04_transforms_2.py
# @author     : zihao lin (quoted from tingsongyu)
# @date       : Oct. 29th 2020 23:19:00
# @brief      : transforms method 2
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
Part 1: Transforms

1. Pad

    transforms.Pad(padding,
                   fill=0,
                   padding_mode='constant')
    
    Function: fill the edge of the picture
    
    :parameter
        a. padding: set the size of fill
           when padding = a, fill the four edges a pixels
           when padding = (a, b), fill b pixels up and down, fill a pixels left and right
           when padding = (a, b, c, d), fill a, b, c, d, pixels left, up, right, down seperately
        b. padding_mode: the mode of filling, constant, edge, reflect, and symmetric
        c. fill: constant, set the value of filled pixel, (R, G, B) or (Gray)
        
2. ColorJitter

    transforms.ColorJitter(brightness=0,
                           contrast=0,
                           saturation=0,
                           hue=0)
                           
    Function: adjust brightness, contrast, saturation and hue
    
    :parameter
        a. brightness: the factor of brightness
           when brightness = a, randomly choose from [max(0, 1-a), 1+a]
           when brightness = (a, b), randomly choose from [a. b]
        b. contrast: the factor of contrast, same as brightness
        c. saturation: the factor of saturation, same as brightness
        d. hue: the factor of hue
           when hue = a, randomly choose from [-a, a], here 0 <= a <= 0.5
           when hue = (a, b), randomly choose from [a, b], here -0.5 <= a <= b <= 0.5
           
3. Grayscale

    Grayscale(num_output_channels)
    
4. RandomGrayscale

    RandomGrayscale(num_output_channels,
                    p=0.1)

    Function: transform the picture into grey picture by some probability
    
    :parameter
        a. num_output_channels: the number of output channels, only 1 or 3
        b. p: the value of the probability of the picture transformed into grey picture

5. RandomAffine

    RandomAffine(degrees,
                 translate=None,
                scale=None,
                shear=None,
                resample=False,
                fillcolor=0)
                
    Function: do the affine transformation to the picture. 
              Affine is a two dimensional linear transformation which formed by five fundamental atom transformations:
              rotate, flip, zoom, translation and shear
              
    :parameter
        a. degrees: the degree of rotate
        b. translate: the interval of translation, (a, b), a is the width, b is the height
           the interval of the translation of the picture in the horizontal dimension is 
                -img_width * a < dx < img_width * a
        c. scale: the proportion of zoom, the unit is area
        d. fill_color: the filled color
        e. shear: the degree of shear, vertical or horizontal
           when shear = a, shear in x axis, the degree of shear is in interval (-a, a)
           when shear = (a, b), shear a degree in x axis and shear b degree in y axis
           when shear = (a, b, c, d), shear degree among (a, b) in x axis, and shear degree among (c, d) in y axis
        f. resample: the resample method, NEAREST, BILINEAR,  or BICUBIC
        
6. RandomErasing

    RandomErasing(p=0.5,
                  scale=(0.02, 0.33),
                  ratio=(0.3, 0.33),
                  value=0,
                  inplace=False)
                  
    Function: erase the picture randomly
    
    :parameter
        a. p: the probability of erasing
        b. scale: the area of erasing
        c. ratio: the ratio of length of width of erasing
        d. value: the value of pixel of erasing, (R, G, B) or (Grey)

7. Lambda

    transforms.Lambda(lambd)
    
    Function: user-defined transforms
    
    :parameter
        lambd: lambda anonymous function
               lambda [arg1 [,arg2, ..., argn]] : expression
               

Part 2: process of transforms

1. transforms.RandomChoice

    transforms.RandomChoice([transforms1, transforms2, transforms3])
    
    Function: randomly choose one transform
    
2. transforms.RandomApply

    transforms.RandomApply([transforms1, transforms2, transforms3])
    
    Function: apply a list of transforms by probability
    
3. transforms.RandomOrder

    transforms.RandomOrder([transforms1, transforms2, transforms3])
    
    Function: shuffle a list of transforms
    
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
split_dir = os.path.abspath(os.path.join("..", "..", "data", "rmb_split"))
if not os.path.exists(split_dir):
    raise Exception(r"data {} does not exists, please go back to lesson02\01_RMB_classification_data.py \
                        to generate the data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # 3 Grayscale
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),

    # 1 RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    # transforms.RandomOrder([transforms.RandomRotation(15),
    #                         transforms.Pad(padding=32),
    #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

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

        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
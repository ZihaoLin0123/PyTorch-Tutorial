import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tools.my_dataset import RMBDataset

'''
1. transforms.CenterCrop

    Function: cut the picture from center
    
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
'''




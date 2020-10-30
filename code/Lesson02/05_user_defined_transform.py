# -*- coding: utf-8 -*-
"""
# @file name  : 05_user_defined_transform.py
# @author     : zihao lin (quoted from tingsongyu)
# @date       : Oct. 30th 2020 00:27:00
# @brief      : user-defined transform
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

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
Two important points:

1. user-defined transforms only receive one parameter and return one parameter
2. pay attention to the output of preview step and the input of the next step

class Compose(object):
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
        
class YourTransforms(object):
    def __init__(self, ...):
        ...
    def __call__(self, img):
        ...
        return img
'''


set_seed(1)  # set random seed

# set parameters
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


class AddPepperNoise(object):
    """add salt and pepper noise
    Args:
        snr （float）: Signal Noise Rate
        p (float): the value of probability
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # salt noise
            img_[mask == 2] = 0     # pepper noise
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# ============================ step 1/5 data ============================
split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "rmb_split"))
if not os.path.exists(split_dir):
    raise Exception(r"data {} does not exists, please go back to lesson02\01_RMB_classification_data.py \
                        to generate the data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AddPepperNoise(0.9, p=0.5),
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

# create DataLoader
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
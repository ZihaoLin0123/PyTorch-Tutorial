import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for RMB classification
        :param data_dir: str, the path of the dataset
        :param transform: torch.transform, preprocess the data
        """

        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)
        # data_info stores the paths and labels of all pictures use index to read sample from DataLoader
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255

        if self.transform is not None:
            img = self.transform(img)  # do some transformation, such as transforming to tensor, etc.

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info

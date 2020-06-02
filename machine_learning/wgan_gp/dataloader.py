import os
import glob
import math
import numpy as np

import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_path_list):
        self.img_path_list = img_path_list
        

    def __len__(self):
        return len(self.img_path_list)
    
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img_npy = np.array(img) / 255 * 2 - 1
        img_npy = img_npy.transpose(2, 0, 1)
        img_tensor = torch.tensor(img_npy)
        return img_tensor



def sample_z(batch_size, dim_z, device):
    return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()


def sample_from_gen(batch_size, dim_z, gen, device):
    z = sample_z(batch_size, dim_z, device)
    fake = gen(z)
    return fake, z
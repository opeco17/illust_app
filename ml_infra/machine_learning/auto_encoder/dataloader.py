import numpy as np
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, images_path):
        self.img_path_list = images_path
    
    
    def __len__(self):
        return len(self.img_path_list)
    
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img_npy = np.array(img) / 255. * 2 - 1
        img_npy = img_npy.transpose(2, 0, 1)
        img_tensor = torch.tensor(img_npy, dtype=torch.float)
        return img_tensor
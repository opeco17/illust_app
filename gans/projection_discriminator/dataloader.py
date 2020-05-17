import numpy as np
import torch
from PIL import Image
import os

def sample_z(batch_size, dim_z, device):
    return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()

def sample_pseudo_labels(num_classes, batch_size, device):
    pseudo_labels = torch.from_numpy(np.random.randint(low=0, high=num_classes, size=(batch_size)))
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels
    
def sample_from_gen(num_classes, batch_size, dim_z, device, gen):
    z = sample_z(batch_size, dim_z, device)
    pseudo_y = sample_pseudo_labels(num_classes, batch_size, device)
    fake = gen(z, pseudo_y) #It's equal to gen.forward method
    return fake, pseudo_y, z


class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_name_list, img_path_list, image_tag_df, hair_colors):
        self.img_name_list = img_name_list
        self.img_path_list = img_path_list
        self.image_tag_df = image_tag_df
        self.hair_colors = hair_colors
        self.hair_colors_dict = {hair_color: num for num, hair_color in enumerate(self.hair_colors)}
        
    
    def __len__(self):
        return len(self.img_path_list)
        
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_name = self.img_name_list[index]
        
        img = Image.open(img_path)
        img_npy = np.array(img)/255*2 - 1
        img_npy = img_npy.transpose(2, 0, 1)
        img_tensor = torch.tensor(img_npy)
        
        label = self.image_tag_df[self.image_tag_df['image name']==img_name]['hair color'].values[0]
        label = self.hair_colors_dict[label]
        label_npy = np.array(label)
        label_tensor = torch.tensor(label_npy)
        
    
        return img_tensor, label_tensor
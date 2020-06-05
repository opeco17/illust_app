import numpy as np
from PIL import Image

import torch

def extract(encoder, img):
    img_npy = np.array(img)/255. * 2 - 1
    img_tensor = torch.tensor(img_npy, dtype=torch.float).view(1, 3, 64, 64)
    feature = encoder(img_tensor).detach().numpy().reshape(-1)
    return feature

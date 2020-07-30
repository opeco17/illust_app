import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model import ResNetGenerator
from sampler import sample_from_gen


def main():
    device = 'cpu'

    hair_colors = ['pink', 'blue', 'brown', 'silver', 'blonde', 'red', 'black', 'white', 'purple']
    hair_colors_dict =  {num: hair_color for num, hair_color in enumerate(hair_colors)}
    
    gen_num_features = 64
    gen_dim_z = 128
    gen_bottom_width = 4
    total_iteration = 200
    num_classes = len(hair_colors)

    gen = ResNetGenerator(
        num_features=gen_num_features, dim_z=gen_dim_z, bottom_width=gen_bottom_width,
        activation=F.relu, num_classes=num_classes,
    )

    gen.load_state_dict(torch.load('./gen_parameter.pth', map_location=torch.device('cpu')))
    gen.eval()

    os.mkdir('./raw_images/') if not os.path.exists('./raw_images/') else None
    for label in hair_colors:
        os.mkdir('./raw_images/{0}'.format(label)) if not os.path.exists('./raw_images/{0}'.format(label)) else None
            
    n_iter = 0
    for i in range(0, num_classes):
        label = hair_colors_dict[i]
        for j in range(total_iteration):
            print(j)
            fake_img, pseudo_y = sample_from_gen(9, 10, 128, i, device, gen)
            for k in range(fake_img.shape[0]):
                fake = (fake_img[k] * 255).astype(np.uint8)
                fake = Image.fromarray(fake)
                fake.save('./raw_images/{0}/{1}.png'.format(label, n_iter))
                n_iter += 1


if __name__ == '__main__':
    main()

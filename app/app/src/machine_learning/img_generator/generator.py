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

    _n_cls = 9
    gen_num_features = 64
    gen_dim_z = 128
    gen_bottom_width = 4
    gen_distribution = 'normal'

    gen = ResNetGenerator(
        gen_num_features, gen_dim_z, gen_bottom_width,
        activation=F.relu, num_classes=_n_cls, distribution=gen_distribution
    )

    gen.load_state_dict(torch.load('./gen_parameter', map_location=torch.device('cpu')))
    gen.eval()

    hair_colors = ['pink', 'blue', 'brown', 'silver', 'blonde', 'red', 'black', 'white', 'purple']
    hair_colors_dict =  {num: hair_color for num, hair_color in enumerate(hair_colors)}
    for label in hair_colors:
        if not os.path.exists('./raw_images/{0}'.format(label)):
            os.mkdir('./raw_images/{0}'.format(label))

    n_iter = 0
    for i in range(0, 9):
        label = hair_colors_dict[i]
        for j in range(200):
            print(j)
            fake_img, pseudo_y = sample_from_gen(9, 10, 128, i, device, gen)
            for k in range(fake_img.shape[0]):
                fake = (fake_img[k] * 255).astype(np.uint8)
                fake = Image.fromarray(fake)
                fake.save('./raw_images/{0}/{1}.png'.format(label, n_iter))
                n_iter += 1


if __name__ == '__main__':
    main()

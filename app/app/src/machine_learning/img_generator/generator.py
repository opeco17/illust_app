import math
import numpy as np
import pandas as pd
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

    n_iter = 0
    for i in range(1000):
        fake_img, pseudo_y = sample_from_gen(9, 32, 128, None, device, gen)
        for j in range(fake_img.shape[0]):
            fake = (fake_img[j] * 255).astype(np.uint8)
            fake = Image.fromarray(fake)
            fake.save('./raw_images/{0}.png'.format(n_iter))
            n_iter += 1


if __name__ == '__main__':
    main()
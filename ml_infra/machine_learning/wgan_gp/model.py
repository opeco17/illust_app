import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


class GeneratorBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None, k_size=3, pad=1, activation=F.relu):
        super(GeneratorBlock, self).__init__()
        
        h_ch = in_ch if h_ch is None else h_ch
        self.activation = activation
        self.c1 = nn.Conv2d(in_ch, h_ch, k_size, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, k_size, 1, pad)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1)
        
        self._initialize()
        
        
    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, gain=1)
        
    
    def forward(self, x):
        h = x
        h = self._upsample(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        out = h + self.c_sc(self._upsample(x))
        return out
    
        
    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')



class ResNetGenerator(nn.Module):
    
    def __init__(self, num_features=64, dim_z = 128, bottom_width=4, activation=F.relu):
        super(ResNetGenerator, self).__init__()
        
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        
        self.l1 = nn.Linear(dim_z, 16*num_features*bottom_width*bottom_width)
        self.block2 = GeneratorBlock(num_features*16, num_features*8, activation=activation)
        self.block3 = GeneratorBlock(num_features*8, num_features*4, activation=activation)
        self.block4 = GeneratorBlock(num_features*4, num_features*2, activation=activation)
        self.block5 = GeneratorBlock(num_features*2, num_features, activation=activation)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

        self._initialize()
        
    
    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv6.weight.data)
          
        
    def forward(self, z):
        h = z
        h = self.l1(h).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.conv6(h)
        h = torch.tanh(h)
        return h



class DiscriminatorBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, k_size=3, pad=1, activation=F.relu, optimize=False):
        super(DiscriminatorBlock, self).__init__()
        
        h_ch = in_ch if h_ch is None else h_ch
    
        self.optimize = optimize
        self.activation = activation
        self.c1 = nn.Conv2d(in_ch, h_ch, k_size, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, k_size, 1, pad)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        
        self._initialize()
        
        
    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)
        
    
    def forward(self, x):
        h = x
        h = self.c1(h) if self.optimize==True else self.c1(self.activation(h))
        h = self.c2(self.activation(h))
        h = self._downsample(h)
        out = h + self._downsample(self.c_sc(x))
        return out
    
    
    def _downsample(self, x):
        return F.avg_pool2d(x, 2)



class ResNetDiscriminator(nn.Module):
    
    def __init__(self, num_features=64, activation=F.relu):
        super(ResNetDiscriminator, self).__init__()
        
        self.num_features = num_features
        self.activation = activation
        
        self.block1 = DiscriminatorBlock(3, num_features, activation=activation, optimize=True)
        self.block2 = DiscriminatorBlock(num_features, num_features*2, activation=activation)
        self.block3 = DiscriminatorBlock(num_features*2, num_features*4, activation=activation)
        self.block4 = DiscriminatorBlock(num_features*4, num_features*8, activation=activation)
        self.block5 = DiscriminatorBlock(num_features*8, num_features*16, activation=activation)
        self.l6 = nn.Linear(num_features*16, 1)
        
        self._initialize()
        
    
    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        
    
    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        h = self.l6(h)
        return h

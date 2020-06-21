import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)
        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


class GeneratorBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, k_size=3, pad=1, activation=F.relu, num_classes=0):
        super(GeneratorBlock, self).__init__()

        self.activation = activation
        if h_ch is None:
            h_ch = out_ch

        self.c1 = nn.Conv2d(in_ch, h_ch, k_size, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, k_size, 1, pad)
        self.b1 = CategoricalConditionalBatchNorm2d(num_classes, in_ch)
        self.b2 = CategoricalConditionalBatchNorm2d(num_classes, h_ch)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, gain=1)
            
    def forward(self, x, y, **kwargs):
        return self.c_sc(self._upsample(x)) + self.residual(x, y)

    def residual(self, x, y):
        h = self.b1(x, y)
        h = self.activation(h)
        h = self._upsample(h)
        h = self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class ResNetGenerator(nn.Module):

    def __init__(self, num_features=64, dim_z=128, bottom_width=4, activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 16*num_features*bottom_width*bottom_width)
        self.block2 = GeneratorBlock(num_features*16, num_features*8, activation=activation, num_classes=num_classes)
        self.block3 = GeneratorBlock(num_features*8, num_features*4, activation=activation, num_classes=num_classes)
        self.block4 = GeneratorBlock(num_features*4, num_features*2, activation=activation, num_classes=num_classes)
        self.block5 = GeneratorBlock(num_features*2, num_features, activation=activation, num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv6.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width) 
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = self.conv6(h)
        return torch.tanh(h)
import math
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as optim


# Generator
class ConditionalBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features):
        super(ConditionalBatchNorm2d, self).__init__(num_features=num_features, affine=False)


    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = self.momentum if self.training else 0.0

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training, exponential_average_factor, self.eps)
 
        # expand dimention to 2 if dimention is 1
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()

        # expand dimention to 4 to calculate affine transformation
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        
        return weight * output + bias



class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, affine=False):
        super(CategoricalConditionalBatchNorm2d, self).__init__(num_features=num_features)
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
            

    def forward(self, x, y):
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
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=False)



class ResNetGenerator(nn.Module):

    def __init__(self, num_features=64, dim_z=128, bottom_width=4, activation=F.relu, num_classes=0):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation

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


    def forward(self, z, y=None):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width) 
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.conv6(h)
        return torch.tanh(h)



# Discriminator
class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, activation=F.relu):
        super(DiscriminatorBlock, self).__init__()
        
        self.activation = activation
        h_ch = in_ch if h_ch is None else out_ch
            
        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        
        self._initialize() 
        

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)
    

    def forward(self, x):
        return F.avg_pool2d(self.c_sc(x), 2) + self.residual(x)
    

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        h = F.avg_pool2d(h, 2)
        return h
    


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()


    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)


    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))


    def residual(self, x):
        h = self.c1(x)
        return F.avg_pool2d(self.c2(h), 2)



class SNResNetProjectionDiscriminator(nn.Module):
    
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = DiscriminatorBlock(num_features, num_features*2, activation=activation)
        self.block3 = DiscriminatorBlock(num_features*2, num_features*4, activation=activation)
        self.block4 = DiscriminatorBlock(num_features*4, num_features*8, activation=activation)
        self.block5 = DiscriminatorBlock(num_features*8, num_features*16, activation=activation)
        self.l6 = utils.spectral_norm(nn.Linear(num_features*16, 1))
        self.l_y = utils.spectral_norm(nn.Embedding(num_classes, num_features*16))

        self._initialize()


    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        init.xavier_uniform_(optional_l_y.weight.data)
        

    def forward(self, x, y):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        output += torch.sum(self.l_y(y)*h, dim=1, keepdim=True)
        return output



class DisLoss():
    def __init__(self):
        pass
    
    def dis_hinge(self, dis_fake, dis_real):
        loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. - dis_fake))
        return loss

    def dis_dcgan(self, dis_fake, dis_real):
        loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
        return loss

    def __call__(self, dis_fake, dis_real):
        return self.dis_dcgan(dis_fake, dis_real)
    
    

class GenLoss():
    def __init__(self):
        pass

    def gen_hinge(self, dis_fake, dis_real=None):
        return -torch.mean(dis_fake)
    
    def gen_dcgan(self, dis_fake, dis_real=None):
        loss =  torch.mean(F.softplus(-dis_fake))
        return loss

    def __call__(self, dis_fake, dis_real=None):
        return self.gen_dcgan(dis_fake, dis_real)
        

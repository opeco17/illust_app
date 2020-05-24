import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import init


def cal_gradient_penalty(dis, real_data, fake_data, coef, device):
    alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    
    dis_interpolates = dis(interpolates)
    
    gradients = autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(dis_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * coef
    return gradient_penalty


def cal_loss_dis(dis_fake, dis_real, gradient_penalty):
    return dis_fake - dis_real + gradient_penalty


def cal_loss_gen(dis_fake):
    return -dis_fake


def cal_wasserstein_distance(dis_fake, dis_real):
    return dis_real - dis_fake
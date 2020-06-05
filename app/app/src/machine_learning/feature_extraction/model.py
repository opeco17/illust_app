import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=4, stride=2, pad=1):
        super(EncoderBlock, self).__init__()
        self.c = nn.Conv2d(in_ch, out_ch, k_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        h = x
        h = self.c(h)
        h = self.bn(h)
        h = self.activation(h)
        return h


class Encoder(nn.Module):
    
    def __init__(self, in_ch=3, h_dim=512, num_features=32):
        super(Encoder, self).__init__()
        self.layer1 = EncoderBlock(in_ch, num_features)
        self.layer2 = EncoderBlock(num_features, num_features*2)
        self.layer3 = EncoderBlock(num_features*2, num_features*4)
        self.layer4 = EncoderBlock(num_features*4, num_features*8)
        self.layer5 = nn.Linear(4096, h_dim)
    
    def forward(self, x):
        h = x
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h).view(-1, 4096)
        h = self.layer5(h)
        return h


def load_model(parameter_path):
    params = torch.load(parameter_path)
    encoder_params = {}
    for key, value in params.items():
        if 'encoder' in key:
            encoder_params[key.lstrip('encoder.')] = value

    encoder = Encoder()
    encoder.load_state_dict(encoder_params)
    encoder.eval()
    return encoder 
	

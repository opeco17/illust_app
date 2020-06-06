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


class DecoderBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, k_size=4, stride=2, pad=1):
        super(DecoderBlock, self).__init__()
        self.ct = nn.ConvTranspose2d(in_ch, out_ch, k_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU()
        
        
    def forward(self, z):
        h = z
        h = self.ct(h)
        h = self.bn(h)
        h = self.activation(h)
        return h


class Decoder(nn.Module):
    
    def __init__(self, h_dim=512, num_features=32, bottom_width=4):
        super(Decoder, self).__init__()
        self.num_features = num_features
        self.bottom_width = bottom_width
        self.fc = nn.Linear(h_dim, 5)
        self.layer1 = nn.Sequential(
            nn.Linear(h_dim, 8*num_features*bottom_width*bottom_width),
            nn.ReLU(),
        )
        self.layer2 = DecoderBlock(8*num_features, 4*num_features)
        self.layer3 = DecoderBlock(4*num_features, 2*num_features)
        self.layer4 = DecoderBlock(2*num_features, num_features)
        self.layer5 = DecoderBlock(num_features, 3)
        
        
    def forward(self, z):
        h = z
        h = self.layer1(h).view(-1, 8*self.num_features, self.bottom_width, self.bottom_width)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        return torch.tanh(h)


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
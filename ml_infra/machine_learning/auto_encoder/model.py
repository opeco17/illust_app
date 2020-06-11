import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_ch):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_ch, in_ch//8, 1)
        self.key_conv = nn.Conv2d(in_ch, in_ch//8, 1)
        self.value_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, ch, width, height = x.size()
        
        h = x
        proj_query = self.query_conv(h).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(h).view(batch_size, -1, width*height)
        s = torch.bmm(proj_query, proj_key)
        attention_map = self.softmax(s)
        
        proj_value = self.value_conv(h).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        out = out.view(batch_size, ch, width, height)
        out = x + self.gamma * out
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, pad=1, optimize=False):
        super(EncoderBlock, self).__init__()
        self.optimize = optimize
        self.c1 = nn.Conv2d(in_ch, in_ch, k_size, stride, pad)
        self.c2 = nn.Conv2d(in_ch, out_ch, k_size, stride, pad)
        self.c_sc = nn.Conv2d(in_ch, out_ch, k_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU()
        
        
    def forward(self, x):
        h = x
        h = self.c1(h) if self.optimize==True else self.c1(self.activation(h))
        h = self.bn1(h)
        h = self.c2(self.activation(h))
        h = self.bn2(h)
        h = self._downsample(h)
        out = h + self._downsample(self.c_sc(x))
        return out
    
    
    def _downsample(self, x):
        return F.avg_pool2d(x, 2)


class Encoder(nn.Module):
    
    def __init__(self, in_ch=3, h_dim=512, num_features=32):
        super(Encoder, self).__init__()
        self.layer1 = EncoderBlock(in_ch, num_features, optimize=True)
        self.layer2 = EncoderBlock(num_features, num_features*2)
        self.sa1 = SelfAttention(num_features*2)
        self.layer3 = EncoderBlock(num_features*2, num_features*4)
        self.sa2 = SelfAttention(num_features*4)
        self.layer4 = EncoderBlock(num_features*4, num_features*8)
        self.layer5 = nn.Linear(4096, h_dim)
    
    def forward(self, x):
        h = x
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.sa1(h)
        h = self.layer3(h)
        h = self.sa2(h)
        h = self.layer4(h).view(-1, 4096)
        h = self.layer5(h)
        return h


class DecoderBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, pad=1, optimize=False):
        super(DecoderBlock, self).__init__()
        self.optimize = optimize
        self.c1 = nn.Conv2d(in_ch, in_ch, k_size, stride, pad)
        self.c2 = nn.Conv2d(in_ch, out_ch, k_size, stride, pad)
        self.c_sc = nn.Conv2d(in_ch, out_ch, k_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU()
        
        
    def forward(self, z):
        h = z
        h = self._upsample(h)
        h = self.c1(h) if self.optimize==True else self.c1(self.activation(h))
        h = self.bn1(h)
        h = self.c2(self.activation(h))
        h = self.bn2(h)
        out = h + self.c_sc(self._upsample(z))
        return out
    
    
    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


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
        self.sa1 = SelfAttention(2*num_features)
        self.layer4 = DecoderBlock(2*num_features, num_features)
        self.sa2 = SelfAttention(num_features)
        self.layer5 = DecoderBlock(num_features, 3)
        
        
    def forward(self, z):
        h = z
        h = self.layer1(h).view(-1, 8*self.num_features, self.bottom_width, self.bottom_width)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.sa1(h)
        h = self.layer4(h)
        h = self.sa2(h)
        h = self.layer5(h)
        return torch.tanh(h)


class AutoEncoder(nn.Module):
    
    def __init__(self, h_dim=512):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(h_dim=h_dim)
        self.decoder = Decoder(h_dim=h_dim)
        
        
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
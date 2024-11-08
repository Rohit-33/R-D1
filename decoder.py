import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAEAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32 , channels)
        self.attention = SelfAttention(1, channels)

    def forward(self , x):
        residue  = x
        x = self.groupnorm(x)
        n , c, h, w = x.shape
        x= x.view((n,c,h*w))
        x = x.transpose(-1,-2)
        x = self.attention(x)
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))
        x += residue
        return x
    

class VAEResidualBlock(nn.Module):
    def __init__(self , inChannels , outChannels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32 , inChannels)
        self.conv1 = nn.Conv2d(inChannels, outChannels , kernel_size = 3 , padding=1)
        self.groupnorm2 = nn.GroupNorm(32 , outChannels)
        self.conv2 = nn.Conv2d(outChannels , outChannels, kernel_size=3,padding = 1)

        if inChannels == outChannels:
            self.residualLayer = nn.Identity()
        else :
            self.residualLayer = nn.Conv2d(inChannels, outChannels,kernel_size=1 , padding = 0)
    
    def forward(self , x):
        residue = x
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residualLayer(residue)

class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1, padding =0),
            nn.Conv2d(4 , 512 , kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512,512),
            VAEResidualBlock(512,512),
            VAEResidualBlock(512,512),
            VAEResidualBlock(512 , 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512 , kernel_size=3 , padding=1),
            VAEResidualBlock(512 , 512),
            VAEResidualBlock(512 , 512),
            VAEResidualBlock(512 , 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512 , 512, kernel_size=3 , padding=1),
            VAEResidualBlock(512 , 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256,256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256 , kernel_size=3, padding=1),
            VAEResidualBlock(256 , 128),
            VAEResidualBlock(128 , 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128,3,kernel_size=3, padding=1),
        )
    
    def forward(self , x):

        x /= 0.18215

        for layer in self:
            x = layer(x)

        return x
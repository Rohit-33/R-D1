import torch
from torch import nn, transpose
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self,nEmbed):
        super().__init__()
        self.linear1 = nn.Linear(nEmbed, 4*nEmbed)
        self.linear2 = nn.Linear(4*nEmbed , nEmbed)

    def forward(self , x):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x

class UNETResidualBlock(nn.Module):
    def __init__(self , inChannels , outChannels , nTime = 1280):
        super().__init__()
        self.groupnormFeature = nn.GroupNorm(32 , inChannels)
        self.convFeature = nn.Conv2d(inChannels, outChannels, kernel_size=3 , padding = 1)
        self.linearTime = nn.Linear(nTime, outChannels)

        self.groupnormMerge = nn.GroupNorm(32, outChannels)
        self.convMerge = nn.Conv2d(outChannels , outChannels, kernel_size=3 , padding =1)

        if inChannels == outChannels:
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv2d(inChannels, outChannels, kernel_size= 3 , padding=0)

    def forward(self , feature , time):
        residue = feature
        feature = self.groupnormFeature(feature)
        feature = F.silu(feature)
        feature =  self.convFeature(feature)
        time = F.silu(time)
        time = self.linearTime(time)
        merge = feature + time.unsqueeze(-1).unsqueeze(-1)
        merge = self.groupnormMerge(merge)
        merge = F.silu(merge)
        merge = self.convMerge(merge)
        return merge + self.residualLayer(residue)
    
class UNETAttentionBlock(nn.Module):
    def __init__(self , nHeads : int , nEmbed : int , dContext = 768):
        super().__init__()
        channels=nHeads*nEmbed

        self.groupnorm = nn.GroupNorm(32,channels)
        self.convInput = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(nHeads, channels, inProjBias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(nHeads, channels, dContext,inProjBias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.linearG1 = nn.Linear(channels, 4*2*channels)
        self.linearG2 = nn.Linear( 4*channels , channels)

        self.convOutput = nn.Conv2d(channels, channels, kernel_size= 1 , padding = 0)

    def forward(self , x , context):
        resLong = x
        x = self.groupnorm(x)
        x = self.convInput(x)
        n , c , h , w = x.shape
        x = x.view((n.c.h*w))
        x = x.transpose(-1,-2)

        resShort = x
        x = self.layernorm1(x)
        x = self.attention1(x)
        x += resShort

        resShort = x
        x = self.layernorm2(x)
        x = self.attention2(x,context)
        x += resShort

        resShort=x
        x = self.layernorm3(x)
        x , g = self.linearG1(x).chunk(2,dim = 1)
        x = x* F.gelu(g)
        x = self.linearG2(x)
        x+= resShort
        x += transpose(-1,-2)
        x = x.vieew((n ,c ,h,w))

        return self.convOutput(x) + resLong




class Upsample(nn.Module):
    def __init__(self , channels):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels , kernel_size=3 , padding =1)
    def forward(self ,x):
        x = F.interpolate(x , scale_factor=2, mode='nearest')
        return self.conv(x)


class ChangeSequential(nn.Sequential):
    def forward(self , x , context , time):
        for layer in self:
            if isinstance(layer , UNETAttentionBlock):
                x = layer(x , context)
            elif isinstance(layer , UNETResidualBlock):
                x = layer(x , time)
            else:
                x = layer(x)
        return(x)
    

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            ChangeSequential(nn.Conv2d(4 , 320,kernel_size=3 , padding=1)),
            ChangeSequential(UNETResidualBlock(320,320) , UNETAttentionBlock(8,40)),
            ChangeSequential(UNETResidualBlock(320, 320) , UNETAttentionBlock(8, 40)),
            ChangeSequential(nn.Conv2d(320,320, kernel_size=3 , stride = 2 , padding=1)),
            ChangeSequential(UNETResidualBlock(320 , 640) , UNETAttentionBlock(8, 80)),
            ChangeSequential(UNETResidualBlock(640 , 640), UNETAttentionBlock(8, 80)),
            ChangeSequential(nn.Conv2d(640, 640 , kernel_size=3 , stride=2 , padding=1)),
            ChangeSequential(UNETResidualBlock(640 , 1280), UNETAttentionBlock(8,160)),
            ChangeSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),
            ChangeSequential(nn.Conv2d(1280 , 1280 , kernel_size=3 , stride=2 , padding=1)),
            ChangeSequential(UNETResidualBlock(1280, 1280)),
            ChangeSequential(UNETResidualBlock(1280, 1280)),
        ])
    
        self.bottleneck = ChangeSequential(
            UNETResidualBlock(1280, 1280), 
            UNETAttentionBlock(8, 160), 
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            ChangeSequential(UNETResidualBlock(2560, 1280)),
            ChangeSequential(UNETResidualBlock(2560, 1280)),
            ChangeSequential(UNETResidualBlock(2560, 1280) , Upsample(1280)),
            ChangeSequential(UNETResidualBlock(2560, 1280) , UNETAttentionBlock(8, 160)),
            ChangeSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8,160)),
            ChangeSequential(UNETResidualBlock(1920,1280) , UNETAttentionBlock(8,160), Upsample(1280)),
            ChangeSequential(UNETResidualBlock(1920, 640) , UNETAttentionBlock(8, 80) ),
            ChangeSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),
            ChangeSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), Upsample(640)),
            ChangeSequential(UNETResidualBlock(960, 320) , UNETAttentionBlock(8, 40)),
            ChangeSequential(UNETResidualBlock(640,320), UNETAttentionBlock(8, 40)),
            ChangeSequential(UNETResidualBlock(640, 320),UNETAttentionBlock(8,40)),

        ])

    def forward(self , x , context , time):
        skipConnections = []

        for layers in self.encoders:
            x = layers(x , context , time)
            skipConnections.append(x)
        
        x = self.bottleneck(x , context , time)

        for layers in self.decoders:
            x = torch.cat((x , skipConnections.pop()) , dim=1)
            x = layers(x, context, time)
        
        return x

class UNETOutputLayer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,inChannels)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)

    def forward(self , x):
        x = self.groupnorm(x)
        x= F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.timeEmbedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent, context, time):
        time = self.timeEmbedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output




        
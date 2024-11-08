import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAEAttentionBlock , VAEResidualBlock

class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(

            # {Batch_Size , (input_image_channle)3 , Hieght , Width} -> {Batch_Size , 128, Height , Width}
            nn.Conv2d(3, 128 , kernel_size = 3 , padding =1),
            VAEResidualBlock(128 , 128),
            VAEResidualBlock(128,128),
            nn.Conv2d(128,128 , kernel_size=2 , stride=2 , padding=0),
            VAEResidualBlock(128,256),
            VAEResidualBlock(256,256),
            nn.Conv2d(256,256, kernel_size=3 , stride=2 , padding=0),
            VAEResidualBlock(256,512),
            VAEResidualBlock(512, 512),
            nn.Conv2d(512 , 512 , kernel_size=3 , stride=2 , padding=0),
            VAEResidualBlock(512 , 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512,512),
            nn.GroupNorm(32 , 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3 , padding=1),
            nn.Conv2d(8,8,kernel_size=1,padding=0),
        )

   # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1 
    def forward(self , x, noise):
        
        for layer in self:
            if getattr(layer , 'stride' , None) == (2,2):
                x= F.pad(x , (0,1,0,1))
            
            x = layer(x)
        mean , logVar = torch.chunk(x , 2 , dim = 1)
        logVar = torch.clamp(logVar ,-30 , 20 )
        variance = logVar.exp()
        stdDev = variance.sqrt()

        x = mean + stdDev*noise

        x *= 0.18215

        return x
        
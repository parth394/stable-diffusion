import torch as th
from torch import nn
from torch.nn import functional as F
import math
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: th.Tensor)-> th.Tensor:
        # x: ( Batch_size, feature, height, width)
        residue = x

        n, c, h, w  = x.shape

        # ( Batch_size, feature, height, width) -> ( Batch_size, feature, height * width)
        x = x.view(n, c, h * w)

        # ( Batch_size, feature, height * width) -> ( Batch_size, height * width,  feature)
        x = x.transpose(-1, -2)

        # ( Batch_size, height * width,  feature) -> ( Batch_size, height * width,  feature)
        x = self.attention(x)

        # ( Batch_size, height * width,  feature) -> ( Batch_size, feature, height * width)
        x = x.transpose(-1, -2)

        # ( Batch_size, feature, height * width) -> ( Batch_size, feature, height, width)
        x = x.view(n, c, h, w)

        # ( Batch_size, feature, height, width) -> ( Batch_size, feature, height, width)
        x += residue

        return x



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        #  x: (Batch_Size, in_channels, Height, Width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.siLU(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.siLU(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)




class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/ 8) -> (batch_size, 512, height/8, width/ 8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/ 8) -> (batch_size, 512, height/4, width/ 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/ 4) -> (batch_size, 512, height/2, width/ 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)

        )

    def forward(self, x: th.Tensor)-> th.Tensor:
        # (batch_size, 512, height/8, width/ 8)
        x/= 0.18215

        for module in self:
            x = module(x)
        
        # (batch_size, 3, height, width)
        return x

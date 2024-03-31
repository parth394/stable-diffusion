import torch as th
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> ( Batch_size, 128, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            
            # ( Batch_size, 128, height, width) -> ( Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # ( Batch_size, 128, height, width) -> ( Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, height, width) -> (batch_size, 128, height//2, width//2)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0 ),

            # ( Batch_size, 128, height//2, width//2) -> ( Batch_size, 256, height//2, width//2)
            VAE_ResidualBlock(128, 256),

            # ( Batch_size, 256, height//2, width//2) -> ( Batch_size, 256, height//2, width//2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 512, height//2, width//2) -> (batch_size, 512, height//4, width//4)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0 ),

            # ( Batch_size, 256, height//4, width//4) -> ( Batch_size, 512, height//4, width//4)
            VAE_ResidualBlock(256, 512),

            # ( Batch_size, 512, height//4, width//4) -> ( Batch_size, 512, height//4, width//4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, height//4, width//4) -> (batch_size, 512, height//8, width//8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0 ),

            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),

            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),

            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),

            # # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            VAE_AttentionBlock(512),

            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512), 

            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            nn.GroupNorm(num_groups=32, num_channels=512),
            
            # ( Batch_size, 512, height//8, width//8) -> ( Batch_size, 512, height//8, width//8)
            # This practically works better than relu for this type of tasks
            nn.SiLU(),

            # (Batch_size, 512, height//8, width//8) -> (batch_size, 8, height//8, width//8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1 ),

            # (Batch_size, 8, height//8, width//8) -> (batch_size, 8, height//8, width//8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),


        )

    def  forward(self, x: th.tensor, noise: th.tensor)-> th.tensor:
        # x: (Batch_size, channels, height, width)
        # Noise: (Batch_size, out_channel, height//8, width//8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # ( Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (batch_size, 8, height//8, width//8) -> 2 * (batch_size, 4, height//8, width//8)
        # Divides the tensor in two tensor along the provided tensor
        mean, log_variance = th.chunk(x, 2, dim=1)

        # (batch_size, 4, height//8, width//8) -> (batch_size, 4, height//8, width//8)
        # This will clamp the values between the specified values
        log_variance = th.clamp(log_variance, -30, 20)

        # Remove the log 
        # (batch_size, 4, height//8, width//8) -> (batch_size, 4, height//8, width//8)
        variance = log_variance.exp()

        # convert variance into standard deviation
        # (batch_size, 4, height//8, width//8) -> (batch_size, 4, height//8, width//8)
        stdev = variance.sqrt()

        # Now we need to sample from the derived distribution
        # Z = N(0, 1) -> N(mean, std)
        # x = mean + stdev * Z
        x = mean + stdev * noise

        # Scale the output using a constant
        # This has come from the research
        x *= 0.18215

        return x
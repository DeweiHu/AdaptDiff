import torch
import torch.nn as nn
import numpy as np

import utils
import modules


class res_Unet(nn.Module):

    def __init__(self, ):
        super().__init__()
        input_dim = 1
        down_dim = (16, 32, 64, 128)
        up_dim = (128, 64, 32, 16)
        output_dim = 2  

        self.silu = nn.SiLU()

        # initial projection
        self.initial = nn.Conv2d(input_dim, down_dim[0], kernel_size=3, padding=1)

        # Downwards sequential
        self.encoder_resblk = nn.ModuleList()
        self.encoder_down_sample = nn.ModuleList()

        for i in range(len(down_dim)-1):
            self.encoder_resblk.append(
                modules.residual_block(down_dim[i], down_dim[i+1])
                )
            self.encoder_down_sample.append(
                modules.DownSample(down_dim[i+1], down_dim[i+1])
                )
        
        # Upwards sequential 
        self.decoder_resblk = nn.ModuleList()
        self.decoder_up_sample = nn.ModuleList()

        for i in range(len(up_dim)-1):
            self.decoder_resblk.append(
                modules.residual_block(2 * up_dim[i], up_dim[i+1])
            )
            self.decoder_up_sample.append(
                modules.UpSample(up_dim[i], up_dim[i])
            )

        # final projection to output
        self.final = nn.Conv2d(up_dim[-1], output_dim, 1)
    

    def forward(self, x):
        # initial projection
        x = self.initial(x)

        # U-Net encoder
        features = []
        for i in range(len(self.encoder_resblk)):
            # residual block
            x = self.encoder_resblk[i](x)

            # buffer for features
            features.append(x)

            # downsample
            x = self.encoder_down_sample[i](x)
        
        # U-Net decoder
        for i in range(len(self.decoder_resblk)):
            # upsample
            x = self.decoder_up_sample[i](x)

            # skip connection
            residual_x = features.pop()
            x = torch.cat((x, residual_x), dim=1)

            # residual block
            x = self.decoder_resblk[i](x)

        output = self.final(x)
        return output
    

if __name__ == "__main__":

    model = res_Unet()

    x = torch.randn((10, 1, 256, 256))
    pred = model(x)

    print(pred.shape)

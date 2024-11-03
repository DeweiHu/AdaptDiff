import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

import src.modules as modules

    
class Residual_block(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Residual_block,self).__init__()
        self.align = nn.Conv2d(in_channels = nch_in,
                               out_channels = nch_out,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0)
        self.dualconv = nn.Sequential(
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out),
                nn.ELU(),
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out)
                )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.align(x)
        x1 = self.dualconv(x)
        opt = self.relu(torch.add(x,x1))
        return opt


def trans_down(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )


def trans_up(in_channels, out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )
                
#%%
class res_UNet(nn.Module):
    def __init__(self, nch_enc, nch_in, nch_out):
        super(res_UNet, self).__init__()
    
        self.nch_enc = nch_enc
        self.nch_aug = nch_enc[:]
        self.nch_aug.insert(0, nch_in)
        self.nch_out = nch_out
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            self.encoder.append(Residual_block(self.nch_aug[i],self.nch_aug[i+1]))
            self.td.append(trans_down(self.nch_enc[i],self.nch_enc[i]))
            # decoder & upsample
            self.tu.append(trans_up(self.nch_enc[-1-i],self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2,self.nch_out))
            else:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2,self.nch_aug[-2-i]))
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        latent = x
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.tu[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred, latent
    

class VAE(nn.Module):
    def __init__(self, enh_enc, seg_enc, nch_in, nch_out):
        super(VAE, self).__init__()
        
        self.enh_enc = enh_enc
        self.seg_enc = seg_enc
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.bifurcate = nn.Conv2d(nch_out,2,1,1,0)
        
        self.Enh_Net = res_UNet(self.enh_enc, self.nch_in, self.nch_out)
        self.Seg_Net = res_UNet(self.seg_enc, self.nch_out, 2)
    
    def reparameterize(self, seg):
        latent = self.bifurcate(seg)
        mu = torch.unsqueeze(latent[:,0,:,:],dim=1)
        log_var = torch.unsqueeze(latent[:,1,:,:],dim=1)
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
#        sample = self.sigmoid(torch.log(eps)-torch.log(1-eps)+\
#                              torch.log(seg)-torch.log(1-seg))
        sample = mu+(eps*std)
        return sample
    
    def forward(self, x):
        # Encoder
        Enh_opt, _ = self.Enh_Net(x)           # [batch,channel=1,H,W]
        # Reparameterize
        z = self.reparameterize(Enh_opt)
        # Decoder
        Seg_opt, _ = self.Seg_Net(z)        
        return Enh_opt, Seg_opt
    

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
    

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, 
                                            dtype=torch.float32, 
                                            device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class condition_Unet(nn.Module):

    def __init__(self,):
        super().__init__()
        image_channels = 1
        down_channels = (16, 32, 64, 128, 256)
        up_channels = (256, 128, 64, 32, 16)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.silu = nn.SiLU()
        
        # initial projection
        self.initial = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downwards sequential
        self.encoder_resblk = nn.ModuleList()
        self.encoder_t_linear = nn.ModuleList()
        self.encoder_down_sample = nn.ModuleList()

        for i in range(len(down_channels)-1):
            self.encoder_resblk.append(modules.residual_block(down_channels[i], down_channels[i+1]))
            self.encoder_t_linear.append(nn.Linear(time_emb_dim, down_channels[i+1]))
            self.encoder_down_sample.append(modules.DownSample(down_channels[i+1], down_channels[i+1]))
        
        # Upwards sequential
        self.decoder_spadeblk = nn.ModuleList()
        self.decoder_t_linear = nn.ModuleList()
        self.decoder_up_sample = nn.ModuleList()

        for i in range(len(up_channels)-1):
            self.decoder_spadeblk.append(modules.SPADE_residual_block(1, 2*up_channels[i], up_channels[i+1]))
            self.decoder_t_linear.append(nn.Linear(time_emb_dim, up_channels[i+1]))
            self.decoder_up_sample.append(modules.UpSample(up_channels[i], up_channels[i]))

        # final projection to output
        self.final = nn.Conv2d(up_channels[-1], out_dim, 1)

    
    def forward(self, x_t, x_condition, t):
        # time embedding
        t_embed = self.time_mlp(t)
        # initial projection 
        x = self.initial(x_t)
        
        # U-Net encoder
        features = []
        for i in range(len(self.encoder_resblk)):
            # residual block
            x = self.encoder_resblk[i](x)
            
            # add time embedding
            time_emb = self.silu(self.encoder_t_linear[i](t_embed))
            time_emb = time_emb[(..., ) + (None, ) * 2]
            x = x + time_emb
            
            features.append(x)

            # downsample
            x = self.encoder_down_sample[i](x)
        

        # U-Net decoder
        for i in range(len(self.decoder_spadeblk)):
            # upsample
            x = self.decoder_up_sample[i](x)

            # skip connection
            residual_x = features.pop()
            x = torch.cat((x, residual_x), dim=1)

            # SPADE block
            x = self.decoder_spadeblk[i](x, x_condition)
            
            # add time embedding
            time_emb = self.silu(self.decoder_t_linear[i](t_embed))
            time_emb = time_emb[(..., ) + (None, ) * 2]
            x = x + time_emb

        output = self.final(x)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
SPADE normalization
Inputs:
    seg_nch (int): channel number of the segmentation map
    output_nch (int): channel number of the output (feature channel) 
Output:
    tensor with shape [b, output_nch, h, w] (feature shape)
'''
class SPADE(nn.Module):

    def __init__(self, seg_nch, output_nch):
        super(SPADE, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(output_nch, affine=False)

        self.embed_nch = 32
        self.embedding = nn.Sequential(
            nn.Conv2d(seg_nch, self.embed_nch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gamma = nn.Conv2d(self.embed_nch, output_nch, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(self.embed_nch, output_nch, kernel_size=3, padding=1)


    def forward(self, x, segmap):
        # step 1: normalize the feature x
        x = self.instance_norm(x)

        # step 2: embed and produce scaling and bias from segmentation map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        seg_embed = self.embedding(segmap)
        gamma = self.gamma(seg_embed)
        beta = self.beta(seg_embed)

        # step 3: residual block
        output = x * (1 + gamma) + beta

        return output
    

class SPADE_residual_block(nn.Module):

    def __init__(self, seg_nch, input_nch, output_nch):
        super(SPADE_residual_block, self).__init__()

        hidden_nch = min(input_nch, output_nch)

        self.conv_1 = nn.Conv2d(input_nch, hidden_nch, kernel_size=3, padding=1)
        self.SPADE_norm_1 = SPADE(seg_nch, input_nch)
        
        self.conv_2 = nn.Conv2d(hidden_nch, output_nch, kernel_size=3, padding=1)
        self.SPADE_norm_2 = SPADE(seg_nch, hidden_nch)

        self.activate = nn.GELU()


    def forward(self, x, segmap):
        dx = self.conv_1(self.activate(self.SPADE_norm_1(x, segmap)))
        dx = self.conv_2(self.activate(self.SPADE_norm_2(dx, segmap)))
        output = x + dx
        
        return output


class residual_block(nn.Module):

    def __init__(self, nch_in, nch_out):
        super(residual_block,self).__init__()
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
                nn.GELU(),
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out)
                )
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.align(x)
        x1 = self.dualconv(x)
        opt = self.gelu(torch.add(x,x1))
        return opt
    
    
def trans_down(nch_in, nch_out):
    return nn.Sequential(
            nn.Conv2d(in_channels=nch_in, 
                      out_channels=nch_out, 
                      kernel_size=4,
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(nch_out),
            nn.GELU()
            )
            
            
def trans_up(nch_in,nch_out):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=nch_in, 
                               out_channels=nch_out,
                               kernel_size=4, 
                               stride=2, 
                               padding=1),
            nn.BatchNorm2d(nch_out),
            nn.GELU()
            )


class res_UNet(nn.Module):
    def __init__(self, nch_enc, nch_in, nch_out):
        super(res_UNet,self).__init__()
        
        # (assume input_channel=1)
        self.nch_in = nch_in
        self.nch_enc = nch_enc
        self.nch_aug = (self.nch_in,)+self.nch_enc
        
        # module list
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            self.encoder.append(residual_block(self.nch_aug[i], self.nch_aug[i+1]))
            self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i]))
            # decoder & upsample
            self.tu.append(trans_up(self.nch_enc[-1-i], self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                self.decoder.append(residual_block(self.nch_aug[-1-i]*2, nch_out))
            else:
                self.decoder.append(residual_block(self.nch_aug[-1-i]*2, self.nch_aug[-2-i]))
    
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.tu[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred


if __name__ == "__main__":
    
    segmap = torch.rand([1, 3, 512, 512])
    feature_map = torch.rand([64, 64])

    ds = F.interpolate(segmap, size=feature_map.size(), mode='nearest')
    print(ds.size())


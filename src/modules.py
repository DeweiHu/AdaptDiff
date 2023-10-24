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
        self.SPADE_norm_1 = SPADE(seg_nch, hidden_nch)
        
        self.conv_2 = nn.Conv2d(hidden_nch, output_nch, kernel_size=3, padding=1)
        self.SPADE_norm_2 = SPADE(hidden_nch, output_nch)

        self.activate = nn.GELU()


    def forward(self, x, segmap):
        dx = self.conv_1(self.activate(self.SPADE_norm_1(x, segmap)))
        dx = self.conv_2(self.activate(self.SPADE_norm_2(dx, segmap)))
        output = x + dx
        
        return output




if __name__ == "__main__":
    
    segmap = torch.rand([1, 3, 512, 512])
    feature_map = torch.rand([64, 64])

    ds = F.interpolate(segmap, size=feature_map.size(), mode='nearest')
    print(ds.size())


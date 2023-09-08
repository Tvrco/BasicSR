import torch
import torch.nn as nn
import numpy as np
import math
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
from basicsr.archs.BSRN_arch import ESDB,BSConvU
from basicsr.archs.Upsamplers import PixelShuffleDirect
from basicsr.utils.registry import ARCH_REGISTRY
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Conv_Block(nn.Module):    
    def __init__(self):
        super(_Conv_Block, self).__init__()
        
        self.cov_block = nn.Sequential(
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25),
            nn.GELU(),
            PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=64),
            nn.GELU()
        )
        
    def forward(self, x):  
        output = self.cov_block(x)
        return output 
    
# @ARCH_REGISTRY.register()
class LapSrnMSV3(nn.Module):
    def __init__(self):
        super(LapSrnMSV3, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
  
        self.convt_I2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)  

        self.convt_I3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block)        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):    
        out = self.relu(self.conv_input(x))
        
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2

        convt_F3 = self.convt_F3(convt_F2)
        convt_I3 = self.convt_I3(HR_4x)
        convt_R3 = self.convt_R3(convt_F3)
        HR_8x = convt_I3 + convt_R3

       
        return HR_2x, HR_4x, HR_8x
        
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss
    
if __name__ == "__main__":
    def count_parameters(model):
        total_params = 0
        conv_count = 0
        for name, param in model.named_parameters():
            print(f"{name}".ljust(45), f"{tuple(param.shape)}".ljust(20), f"param:{param.numel():,}")
            if param.requires_grad:
                total_params += param.numel()
        for name,module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                conv_count += 1
        print(f"Total number of trainable parameters: {total_params:,}")
        print(f"Total number of conv2d: {conv_count:,}")
        return total_params

    model = LapSrnMSV3().cuda()
    print(model)
    count_parameters(model)
    # print(count_conv_layers(model))
    summaryv2(model, (1,3,16,16))
    # summaryv1(model,(3,16,16))
    # output_images = model(x)
    # for i, img in enumerate(output_images):
    #     print(f"Layer {i+1}".ljust(5),f": {img.shape}")

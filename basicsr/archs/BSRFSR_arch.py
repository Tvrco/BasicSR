from re import X
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
from basicsr.archs.BSRNre_arch import BSDB,BSConvU,BSConvU_rep
from basicsr.archs.LPE_arch import LightPriorEstimationNetwork
from basicsr.archs.LPEv2_arch import LightPriorEstimationNetwork as LightPriorEstimationNetwork2
from basicsr.archs.Upsamplers import PixelShuffleDirect
from basicsr.utils.registry import ARCH_REGISTRY

def cont(model,x=(3, 16, 16)):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, x, as_strings=True,print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # thop
    from thop import profile
    net_cls_str = f'{model.__class__.__name__}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = torch.randn(1, 3, 16, 16).to(device)
    flops, params = profile(model, (inputs, ))
    print(f'Network: {net_cls_str}, with flops(128 x 128): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')


class _Conv_Block(nn.Module):
    def __init__(self,num_out_ch=3,conv=BSConvU):
        super(_Conv_Block, self).__init__()

        self.B1 = BSDB(in_channels=64, out_channels=64, conv=conv, p=1)
        self.B2 = BSDB(in_channels=64, out_channels=64, conv=conv, p=1)
        self.B3 = BSDB(in_channels=64, out_channels=64, conv=conv, p=1)

        self.conv1 = nn.Conv2d(64 * 3, 64, 1)
        self.up = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.GELU = nn.GELU()

    def forward(self, x):
        # print(x.shape)
        output_B1 = self.B1(x)
        output_B2 = self.B2(output_B1)
        output_B3 = self.B3(output_B2)
        output = torch.cat([output_B1,output_B2,output_B3],dim = 1)
        output = self.conv1(output)
        output = self.GELU(output)
        output = self.up(output)
        return output

@ARCH_REGISTRY.register()
class BSRFSR(nn.Module):
    def __init__(self,num_out_ch=3,conv='BSConvU'):
        super(BSRFSR, self).__init__()
        if conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()

        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64,conv =self.conv)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64,conv =self.conv)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3,conv =self.conv)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB1(convt_SRB1) # 64*16*16 双路
        convt_DLB1 = self.convt_DLB1(convt_SRB1) # 64*32*32
        down_DLB1 = self.down_DLB1(self.relu(convt_DLB1)) # 3*32*32
        up_TRB1 = self.up_TRB1(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(convt_DLB1)
        convt_TRB2 = self.convt_up_TRB2(up_TRB1)
        convt_TRB2 = self.convt_TRB2(convt_TRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(self.relu(convt_DLB2))# 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(convt_DLB2)
        convt_TRB3 = self.convt_up_TRB3(up_TRB2)
        convt_TRB3 = self.convt_TRB3(convt_TRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))
        HR_8x = convt_DLB3 + up_TRB3
        return HR_2x,HR_4x,HR_8x

if __name__ == "__main__":
    model = BSRFSR(num_out_ch=3,conv='BSConvU_rep')
    summaryv2(model, (1,3,16,16))
    cont(model,x=(3, 16, 16))
from re import X
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
from basicsr.archs.BSRN_arch import ESDB,BSConvU
from basicsr.archs.LPE_arch import LightPriorEstimationNetwork
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

class BSRB(nn.Module):
    def __init__(self,d):
        super(BSRB, self).__init__()
        self.conv = BSConvU
        self.block = nn.Sequential()
        # 对于给定的循环次数 d，创建前向传播模块的序列
        for i in range(d): 
            self.block.add_module("GELU_" + str(i), nn.GELU())
            self.block.add_module(
                "BSConvU_" + str(i),
                self.conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))
    # 定义前向传播函数
    def forward(self, x):
        # 应用前向传播模块的序列  
        output = self.block(x)  
        return output


class _Conv_Block(nn.Module):
    def __init__(self,num_out_ch=3):
        super(_Conv_Block, self).__init__()

        self.B1 = ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25)
        self.B2 = ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25)
        self.B3 = ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25)

        self.conv1 = nn.Conv2d(64 * 3, 64, 1)
        self.up = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.GELU = nn.GELU()

    def forward(self, x):
        output_B1 = self.B1(x)
        output_B2 = self.B2(output_B1)
        output_B3 = self.B3(output_B2)
        output = torch.cat([output_B1,output_B2,output_B3],dim = 1)
        output = self.conv1(output)
        output = self.GELU(output)
        output = self.up(output)
        return output

# @ARCH_REGISTRY.register()
class LapSrnMSV4(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB1(convt_SRB1) # 64*16*16 双路
        convt_DLB1 = self.convt_DLB1(convt_SRB1) # 64*32*32
        down_DLB1 = self.down_DLB1(convt_DLB1) # 3*32*32
        up_TRB1 = self.up_TRB1(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+convt_DLB1
        convt_TRB2 = self.convt_TRB2(convt_SRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(convt_DLB2) # 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+convt_DLB2
        convt_TRB3 = self.convt_TRB3(convt_SRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x

# @ARCH_REGISTRY.register()
class LapSrnMSV4_2(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_2, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB1(convt_SRB1) # 64*16*16 双路
        convt_DLB1 = self.convt_DLB1(convt_SRB1) # 64*32*32
        down_DLB1 = self.down_DLB1(convt_DLB1) # 3*32*32
        up_TRB1 = self.up_TRB1(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)
        convt_TRB2 = self.convt_TRB2(convt_SRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(convt_DLB2) # 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)
        convt_TRB3 = self.convt_TRB3(convt_SRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x

   
# @ARCH_REGISTRY.register()
class LapSrnMSV4_3(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_3, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        self.bsrb = BSRB(d=8)
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        # self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.bsrb(out) # 64*16*16
        # convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB1(convt_SRB1) # 64*16*16 双路
        convt_DLB1 = self.convt_DLB1(convt_SRB1) # 64*32*32
        down_DLB1 = self.down_DLB1(convt_DLB1) # 3*32*32
        up_TRB1 = self.up_TRB1(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+convt_DLB1
        convt_TRB2 = self.convt_TRB2(convt_SRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(convt_DLB2) # 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+convt_DLB2
        convt_TRB3 = self.convt_TRB3(convt_SRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x


# @ARCH_REGISTRY.register()
class LapSrnMSV4_4(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_4, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        self.bsrb = BSRB(d=8)
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        # self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB2 = BSConvU(3, 64, kernel_size=3, **kwargs)
        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(3, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.bsrb(out) # 64*16*16
        # convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB1(convt_SRB1) # 64*16*16 双路
        up_TRB1 = self.up_TRB1(self.relu(convt_TRB1)) # 3*32*32

        convt_DLB1 = self.convt_DLB1(convt_SRB1)+up_TRB1 # 64*32*32
        down_DLB1 = self.down_DLB1(convt_DLB1) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+convt_DLB1
        convt_TRB2 = self.convt_TRB2(convt_SRB2)
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))

        convt_DLB2 = self.convt_DLB2(convt_SRB2)+up_TRB2
        down_DLB2 = self.down_DLB2(convt_DLB2) # 3*32*32
        HR_4x = down_DLB2 

        convt_SRB3 = self.convt_SRB3(HR_4x)+convt_DLB2
        convt_TRB3 = self.convt_TRB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))

        convt_DLB3 = self.convt_DLB3(convt_SRB3)+up_TRB3

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x


# @ARCH_REGISTRY.register()
class LapSrnMSV4_5(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_5, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)
   

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

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
        convt_TRB2 = self.convt_TRB2(convt_SRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(self.relu(convt_DLB2))# 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(convt_DLB2)
        convt_TRB3 = self.convt_TRB3(convt_SRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x

# @ARCH_REGISTRY.register()
class LapSrnMSV4_6(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_6, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)

        self.esdb_conv = _Conv_Block(num_out_ch=64)
        self.donw_dlb = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)



    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB(convt_SRB1) # 64*16*16 双路
        esdb_conv1 = self.esdb_conv(convt_SRB1) # 64*32*32
        down_DLB1 = self.donw_dlb(self.relu(esdb_conv1)) # 3*32*32
        up_TRB1 = self.up_TRB(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(esdb_conv1)
        convt_TRB2 = self.convt_TRB(convt_SRB2)
        esdb_conv2 = self.esdb_conv(convt_SRB2)
        down_DLB2 = self.donw_dlb(self.relu(esdb_conv2))# 3*32*32
        up_TRB2 = self.up_TRB(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(esdb_conv2)
        convt_TRB3 = self.convt_TRB(convt_SRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x

# @ARCH_REGISTRY.register()
class LapSrnMSV4_7(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_7, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)

        self.esdb_conv = _Conv_Block(num_out_ch=64)
        self.donw_dlb = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.conv1 = nn.Conv2d(64 , 3, 1)


    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        convt_TRB1 = self.convt_TRB(convt_SRB1) # 64*16*16 双路
        esdb_conv1 = self.esdb_conv(convt_SRB1) # 64*32*32
        down_DLB1 = self.donw_dlb(self.relu(esdb_conv1)) # 3*32*32
        up_TRB1 = self.up_TRB(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(esdb_conv1)
        convt_TRB2 = self.convt_TRB(convt_SRB2)
        esdb_conv2 = self.esdb_conv(convt_SRB2)
        down_DLB2 = self.donw_dlb(self.relu(esdb_conv2))# 3*32*32
        up_TRB2 = self.up_TRB(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(esdb_conv2)
        convt_TRB3 = self.convt_TRB(convt_SRB3)
        convt_DLB3 = self.esdb_conv(convt_SRB3)
        down_DLB3 = self.conv1(convt_DLB3)
        up_TRB3 = self.up_TRB(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = down_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x


# @ARCH_REGISTRY.register()
class LapSrnMSV4_8(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_8, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)

        self.esdb_conv = _Conv_Block(num_out_ch=64)
        self.donw_dlb = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.convt_UPB = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)

        self.conv1 = nn.Conv2d(64 , 3, 1)


    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        x_t = convt_SRB1.clone()

        convt_TRB1 = self.convt_TRB(x_t) # 64*16*16 双路
        esdb_conv1 = self.esdb_conv(convt_SRB1) # 64*32*32
        down_DLB1 = self.donw_dlb(self.relu(esdb_conv1)) # 3*32*32
        up_TRB1 = self.up_TRB(self.relu(convt_TRB1)) # 3*32*32
        # print(down_DLB1.shape)
        # print(up_TRB1.shape)
        HR_2x = down_DLB1 + up_TRB1
        # print(HR_2x.shape)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(esdb_conv1)
        convt_TRB2 = self.convt_TRB(self.convt_UPB(up_TRB1))
        esdb_conv2 = self.esdb_conv(convt_SRB2)
        down_DLB2 = self.donw_dlb(self.relu(esdb_conv2))# 3*32*32
        up_TRB2 = self.up_TRB(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(esdb_conv2)
        convt_TRB3 = self.convt_TRB(self.convt_UPB(up_TRB2))
        convt_DLB3 = self.esdb_conv(convt_SRB3)
        down_DLB3 = self.conv1(convt_DLB3)
        up_TRB3 = self.up_TRB(self.relu(convt_TRB3))

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = down_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x
    
# @ARCH_REGISTRY.register()
class LapSrnMSV4_9(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_9, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

    def forward(self, x):
        out = self.relu(self.fea_conv(x))
        convt_SRB1 = self.convt_SRB1(out) # 64*16*16
        x_t_clone = convt_SRB1.clone()


        convt_TRB1 = self.convt_TRB1(x_t_clone) # 64*16*16 双路
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

        # print(f'convt_SRB3:{convt_SRB3.shape}')
        # print(f'convt_TRB3:{convt_TRB3.shape}')
        # print(f'convt_DLB3:{convt_DLB3.shape}')
        # print(f'up_TRB3:{up_TRB3.shape}')
        HR_8x = convt_DLB3 + up_TRB3
        # print(HR_8x.shape)
        return HR_2x,HR_4x,HR_8x

# @ARCH_REGISTRY.register()
class LapSrnMSV4_10(nn.Module):
    def __init__(self,num_out_ch=3):
        super(LapSrnMSV4_10, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)

        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)


        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

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

class LPEB(nn.Module):
    def __init__(self,dim=128,num_out_ch=3):
        super(LPEB, self).__init__()
        kwargs = {'padding': 1}
        self.LPE = LightPriorEstimationNetwork(dim=dim)
        self.esdb_1_1 = ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25)
        self.esdb_1_2 = ESDB(in_channels=64, out_channels=64, conv=BSConvU, p=0.25)
        self.convt_SRB1_2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_SRB1_3 = BSConvU(64+11, 64, kernel_size=3, **kwargs)
        self.convt_SRB1_4 = BSConvU(64, 3, kernel_size=3, **kwargs)

    def forward(self,x):
        LPE_HR_2x = self.LPE(x) #11*32*32
        convt_SRB1_2 = self.convt_SRB1_2(x)
        esdb1 = self.esdb_1_1(convt_SRB1_2)

        HR_2x_1 = torch.cat((esdb1,LPE_HR_2x),dim=1)
        esdb_1_2 = self.esdb_1_2(self.convt_SRB1_3(HR_2x_1))
        sr = self.convt_SRB1_4(esdb_1_2)
        return sr,esdb_1_2,LPE_HR_2x


@ARCH_REGISTRY.register()
class LapSrnMSV4_11(nn.Module):
    def __init__(self,num_out_ch=3,dim=128):
        super(LapSrnMSV4_11, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)
        self.LPEB_1 = LPEB(dim=dim)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)
        self.LPEB_2 = LPEB(dim=dim)


        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.LPEB_3 = LPEB(dim=dim)
        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

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
        HR_2x,esdb_1_2,fb_sr2 = self.LPEB_1(HR_2x)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(esdb_1_2)
        convt_TRB2 = self.convt_up_TRB2(up_TRB1)
        convt_TRB2 = self.convt_TRB2(convt_TRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(self.relu(convt_DLB2))# 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2
        HR_4x,esdb_1_2,fb_sr4 = self.LPEB_2(HR_4x)

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(esdb_1_2)
        convt_TRB3 = self.convt_up_TRB3(up_TRB2)
        convt_TRB3 = self.convt_TRB3(convt_TRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))
        HR_8x = convt_DLB3 + up_TRB3
        HR_8x,_,fb_sr8 = self.LPEB_3(HR_8x)
        return HR_2x,HR_4x,HR_8x,fb_sr2,fb_sr4,fb_sr8

@ARCH_REGISTRY.register()
class LapSrnMSV4_12(nn.Module):
    def __init__(self,num_out_ch=3,dim=64):
        super(LapSrnMSV4_12, self).__init__()
        kwargs = {'padding': 1}
        self.fea_conv = BSConvU(in_channels=3, out_channels=64, kernel_size=3, **kwargs)
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.GELU()
        
        self.convt_SRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_DLB1 = _Conv_Block(num_out_ch=64)
        self.down_DLB1 = BSConvU(64, 3, kernel_size=3, **kwargs)
        self.LPEB = LPEB(dim=dim)

        self.convt_TRB1 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.up_TRB1 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        self.convt_SRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_DLB2 = _Conv_Block(num_out_ch=64)
        self.down_DLB2 = BSConvU(64, 3, kernel_size=3, **kwargs)
        # self.LPEB_2 = LPEB(dim=dim)


        self.convt_TRB2 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB2 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB2 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=num_out_ch)

        # self.LPEB_3 = LPEB(dim=dim)
        self.convt_SRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.convt_TRB3 = BSConvU(64, 64, kernel_size=3, **kwargs)
        self.convt_up_TRB3 = BSConvU(num_out_ch, 64, kernel_size=3, **kwargs)
        self.up_TRB3 = PixelShuffleDirect(scale=2, num_feat=64, num_out_ch=3)
        self.convt_DLB3 = _Conv_Block(num_out_ch=3)

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
        HR_2x,esdb_1_2,fb_sr2 = self.LPEB(HR_2x)

        convt_SRB2 = self.convt_SRB2(HR_2x)+self.relu(esdb_1_2)
        convt_TRB2 = self.convt_up_TRB2(up_TRB1)
        convt_TRB2 = self.convt_TRB2(convt_TRB2)
        convt_DLB2 = self.convt_DLB2(convt_SRB2)
        down_DLB2 = self.down_DLB2(self.relu(convt_DLB2))# 3*32*32
        up_TRB2 = self.up_TRB2(self.relu(convt_TRB2))
        HR_4x = down_DLB2 + up_TRB2
        HR_4x,esdb_1_2,fb_sr4 = self.LPEB(HR_4x)

        convt_SRB3 = self.convt_SRB3(HR_4x)+self.relu(esdb_1_2)
        convt_TRB3 = self.convt_up_TRB3(up_TRB2)
        convt_TRB3 = self.convt_TRB3(convt_TRB3)
        convt_DLB3 = self.convt_DLB3(convt_SRB3)
        up_TRB3 = self.up_TRB3(self.relu(convt_TRB3))
        HR_8x = convt_DLB3 + up_TRB3
        HR_8x,_,fb_sr8 = self.LPEB(HR_8x)
        return HR_2x,HR_4x,HR_8x,fb_sr2,fb_sr4,fb_sr8

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


'''
验证model
'''
# if __name__ == "__main__":
#     def count_parameters(model):
#         total_params = 0
#         conv_count = 0
#         for name, param in model.named_parameters():
#             print(f"{name}".ljust(45), f"{tuple(param.shape)}".ljust(20), f"param:{param.numel():,}")
#             if param.requires_grad:
#                 total_params += param.numel()
#         for name,module in model.named_modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
#                 conv_count += 1
#         print(f"Total number of trainable parameters: {total_params:,}")
#         print(f"Total number of conv2d: {conv_count:,}")
#         return total_params

#     model = LapSrnMSV4_12(num_out_ch=3)
#     # model2 = LPEB()
#     # print(model)
#     # count_parameters(model)
#     # print(count_conv_layers(model))
#     summaryv2(model, (1,3,16,16))
#     # summaryv1(model,(3,16,16))
#     # output_images = model(torch.randn((2, 3, 16, 16)))
#     # for i, img in enumerate(output_images):
#     #     print(f"Layer {i+1}".ljust(5),f": {img.shape}")

'''
inference
'''

if __name__ == '__main__':
    import argparse
    import cv2
    import glob
    import numpy as np
    import os
    import torch
    from tqdm import tqdm
    from basicsr.utils.img_util import img2tensor, tensor2img
    import torchvision
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='datasets/data/inference')
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/LapSRNV4.11_Celeb26k_BS64_L1_600k/models/net_g_100000.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path)
    result_root = f'results/fbsr_result/{os.path.basename(args.test_path)}'
    os.makedirs(result_root, exist_ok=True)

    # set up the LapSrnMSV
    net = LapSrnMSV4_11(num_out_ch=3,dim=64).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    net.eval()

    # scan all the jpg and png images
    img_list = sorted(glob.glob(os.path.join(test_root, '*.[jp][pn]g')))
    print(img_list)
    pbar = tqdm(total=len(img_list), desc='')
    for idx, img_path in enumerate(img_list):
        img_name = os.path.basename(img_path).split('.')[0]
        pbar.update(1)
        pbar.set_description(f'{idx}: {img_name}')
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        # inference
        s = time.time()
        with torch.no_grad():
            HR_2x,HR_4x,output,fb_sr2,fb_sr4,fb_sr8 = net(img)
            e = time.time()
            print(f'inference time{e-s}')
        # save image
        output = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
        save_img_path = os.path.join(result_root, f'{img_name}_bfsr_sr8.png')
        cv2.imwrite(save_img_path, output)
        print(fb_sr8.shape)
        a,b=fb_sr8.shape[2],fb_sr8.shape[3]
        reshaped_tensor  = fb_sr8.view(11, 1, 128,128)
        torchvision.utils.save_image(reshaped_tensor, f'{result_root}/fb_{img_name}_{a}_{b}.png', nrow=11)
        from torchvision import transforms
        import matplotlib.pyplot as plt
        t = transforms.ToPILImage()
        fig = plt.figure()
        for i in range(11):
            # 将numpy数组转换为图像对象。
            face_img_tensor = fb_sr8[0,i,:,:]
            _img = t(face_img_tensor)
            fig.add_subplot(4, 3,i+1)
            # vmin=0, vmax=255表示图像像素值的范围是0到255。
            plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
            # plt.imshow(pmaps[:,:,i])
        plt.savefig(f'{result_root}/fb_{img_name}_{a}_{b}_all.png')
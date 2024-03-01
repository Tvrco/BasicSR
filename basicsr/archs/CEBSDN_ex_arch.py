'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs.CA_arch import CoordAtt
from basicsr.archs.CEBSDN_arch import ESA
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1

class CA(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(CA, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = nn.Conv2d(in_ch, in_ch, 1)
        self.ca = CoordAtt(in_ch, reduction=32)
        self.act = nn.GELU()

    def forward(self, input):
        out = self.act(self.pw(input))
        out =self.ca(out)
        out = self.dw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        fea = self.pw1(x)
        fea = self.pw2(fea)
        fea = self.dw(fea)
        return fea

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CADB_wo_CA(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(CADB_wo_CA, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c3_5 = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c12_cat = nn.Conv2d(self.dc * 3, in_channels, 1)

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        # self.ca = CA(in_channels, in_channels)
        # self.ca = CoordAtt(in_channels, reduction=32)
        # self.esa = ESA(in_channels, conv)
        # self.cca = CCALayer(in_channels)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input)) # 32
        r_c1 = (self.c1_r(input)) # 64
        r_c1 = self.act(r_c1 + input) # 64

        distilled_c2 = self.act(self.c2_d(r_c1)) # 32
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)
        r_c3_5 = self.act(self.c3_5(r_c2))
        

        cat_1 = torch.cat([distilled_c1, distilled_c2,r_c3_5], dim=1) #32 ,32 ,64
        cat_1_c12_cat = (self.c12_cat(cat_1))
        cat_1_c12_cat = self.act(cat_1_c12_cat + r_c2)

        distilled_c3 = self.act(self.c3_d(cat_1_c12_cat))
        r_c3 = (self.c3_r(cat_1_c12_cat))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        # print(f"distilled_c1:{distilled_c1.shape}") # ([1, 32, 64, 64])
        # print(f"distilled_c2:{distilled_c2.shape}")
        # print(f"r_c2:{r_c2.shape}")
        # print(f"cat_1:{cat_1.shape}") # ([1, 96, 64, 64])
        # print(f"cat_1_c12_cat:{cat_1_c12_cat.shape}") # ([1, 64, 64, 64])
        # print(f"out:{out.shape}")
        out = self.c5(out)
        # out_fused = self.ca(out)
        # out_fused = self.esa(out)
        # out_fused = self.cca(out_fused)
        return out + input

class CADB_wo_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=1):
        super(CADB_wo_Cat, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = BSConvU(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.ca = CA(in_channels, in_channels)


    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        # print(f'out:{out.shape}')
        out_fused = self.ca(out)
        # print(out_fused.shape)

        return out_fused + input

@ARCH_REGISTRY.register()
class CEBSDN_wo_CA(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
        super(CEBSDN_wo_CA, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        # print(conv)
        if conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvS':
            self.conv = BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = CADB_wo_CA(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        # print('-'*50)
        # print(out_lr.shape)
        output = self.upsampler(out_lr)

        return output

@ARCH_REGISTRY.register()
class CEBSDN_wo_Cat(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
        super(CEBSDN_wo_Cat, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        # print(conv)
        if conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvS':
            self.conv = BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = CADB_wo_Cat(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        # print('-'*50)
        # print(out_lr.shape)
        output = self.upsampler(out_lr)

        return output
if __name__ == "__main__":
    # model = CEBSDN_wo_CA(upscale=8)
    # FLOPs: 117.866496M, FLOPs: 0.117866496G,Params: 464.64K
    model = CEBSDN_wo_Cat(upscale=8)
    # FLOPs: 111.198208M, FLOPs: 0.111198208G,Params: 450.112K

    # summaryv2(model, (1,3,16,16))
    # summaryv1(model,(3,128,128))
    # print(model)

    input_data = torch.randn((1, 3, 16, 16))
    output_data = model(input_data)
    print(output_data.shape)
    macs, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {macs / 1e6}M, FLOPs: {macs / 1e9}G,Params: {params / 1e3}K")

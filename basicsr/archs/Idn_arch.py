
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                    padding=0, bias=True)
class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU,
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU,
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax,
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, upsample_type=None,
                    kernel_size=None,):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor + 1
            padding = (kernel_size - 1) // 2
            output_padding = scale_factor - 1
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                stride=scale_factor, padding=padding,
                                                output_padding=output_padding, bias=True)
        elif upsample_type == 'pixelshuffle':
            ks = kernel_size if kernel_size is not None else 3
            padding = (ks - 1) // 2
            self.up_conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels * (scale_factor**2), ks, 1, padding),
                                nn.PixelShuffle(scale_factor)
                            )
        else:
            ks = kernel_size if kernel_size is not None else 3
            padding = (ks - 1) // 2
            self.up_conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, ks, 1, padding),
                                nn.Upsample(scale_factor=scale_factor, mode='bicubic')
                            )

    def forward(self, x):
        return self.up_conv(x)


# Regular convolution -> activation
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                    groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **kwargs)
        )
@ARCH_REGISTRY.register()
class IDN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, num_blocks=4, D3=64, s=4,
                    act_type='leakyrelu', upsample_type='deconvolution'):
        super(IDN, self).__init__()
        assert s > 1, 's should be larger than 1, otherwise split_ratio will be out of range.\n'
        split_ratio = 1 / s
        d = int(split_ratio * D3)
        self.upscale = upscale

        self.fblock = nn.Sequential(
                            ConvAct(in_channels, D3, 3, act_type=act_type),
                            ConvAct(D3, D3, 3, act_type=act_type)
                        )

        layers = []
        for i in range(num_blocks):
            layers.append(DBlock(D3, d, act_type))
        self.dblocks = nn.Sequential(*layers)

        self.rblock = Upsample(D3, out_channels, upscale, upsample_type, 17)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')

        x = self.fblock(x)
        x = self.dblocks(x)
        x = self.rblock(x)

        x += x_up

        return x


class DBlock(nn.Sequential):
    def __init__(self, D3, d, act_type):
        super(DBlock, self).__init__(
            EnhancementUnit(D3, d, act_type),
            conv1x1(D3 + d, D3)
        )


class EnhancementUnit(nn.Module):
    def __init__(self, D3, d, act_type, groups=[1,4,1,4,1,1]):
        super(EnhancementUnit, self).__init__()
        assert len(groups) == 6, 'Length of groups should be 6.\n'
        self.d = d

        self.conv1 = nn.Sequential(
                            ConvAct(D3, D3 - d, 3, groups=groups[0], act_type=act_type),
                            ConvAct(D3 - d, D3 - 2*d, 3, groups=groups[1], act_type=act_type),
                            ConvAct(D3 - 2*d, D3, 3, groups=groups[2], act_type=act_type),
                        )

        self.conv2 = nn.Sequential(
                            ConvAct(D3 - d, D3, 3, groups=groups[3], act_type=act_type),
                            ConvAct(D3, D3 - d, 3, groups=groups[4], act_type=act_type),
                            ConvAct(D3 - d, D3 + d, 3, groups=groups[5], act_type=act_type),
                        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x_c = x[:, :self.d, :, :]
        x_c = torch.cat([x_c, residual], dim=1)
        x_s = x[:, self.d:, :, :]
        x_s = self.conv2(x_s)

        return x_s + x_c
if __name__ == "__main__":
    model = IDN(3,3,8)
    # summaryv2(model, (1,3,16,16))
    # summaryv1(model,(3,128,128))
    # print(model)

    input_data = torch.randn((1, 3, 16, 16))
    output_data = model(input_data)
    print(output_data.shape)
    macs, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {macs / 1e6}M, FLOPs: {macs / 1e9}G,Params: {params / 1e3}K")

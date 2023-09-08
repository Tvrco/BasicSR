import torch
import torch.nn as nn
import torchvision
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
from basicsr.archs.BSRN_arch import ESDB,BSConvU

class Residual(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        hdim = int(outs/2)
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(True),
            nn.Conv2d(ins, hdim, 1),
            nn.BatchNorm2d(hdim),
            nn.ReLU(True),
            BSConvU(hdim, hdim, 3, 1, 1),
            nn.BatchNorm2d(hdim),
            nn.ReLU(True),
            nn.Conv2d(hdim, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            BSConvU(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            BSConvU(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.layers(x)
        return out


class BSHB(nn.Module):
    def __init__(self, dim, n, ):
        super(BSHB, self).__init__()
        self._dim = dim
        self._n = n
        self._init_layers(self._dim, self._n)

    def _init_layers(self, dim, n):
        setattr(self, 'res' + str(n) + '_1', Residual(dim, dim))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n - 1)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self, 'res' + str(n) + '_3', Residual(dim, dim))
        setattr(self, 'unsample' + str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n - 1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'unsample' + str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)

class LightPriorEstimationNetwork(nn.Module):

    def __init__(self,dim):
        super(LightPriorEstimationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            Residual(64, dim),
            ResBlock(dim),
            # ResBlock(128),
        )
        self.hg_blocks = nn.Sequential(
            BSHB(dim, 3),
            BSHB(dim, 3),
        )
        self.con_block = nn.Sequential(

            nn.Conv2d(dim, 11, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(11), 
            nn.ReLU(True),
            )
    def forward(self, x):
        # print(f"input_shape:{x.shape}")
        out = self.conv1(x)
        # print(f"conv1_shape:{out.shape}")
        out = self.res_blocks(out)
        # print(f"res_blocks_shape:{out.shape}")
        out = self.hg_blocks(out)
        # print(f"hg_out_shape:{out.shape}")
        out = self.con_block(out)
        return out
def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)
if __name__ == "__main__":
    x = torch.randn((1, 3, 32, 32))
    model = LightPriorEstimationNetwork(128)
    b = model(x)
    # print(b.shape)
    # model = ResBlock(128)
    summaryv2(model, (1,3,32,32))
    # summaryv1(model,(3,32,32))
    # print_network(model)
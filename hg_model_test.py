from torch import nn as nn
import torch
from torch.nn import functional as F
from thop import profile,clever_format
import time
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
            nn.Conv2d(hdim, hdim, 3, 1, 1),
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


class HourGlassBlock(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, dim, n):
        super(HourGlassBlock, self).__init__()
        self._dim = dim
        self._n = n
        self._init_layers(self._dim, self._n)

    def _init_layers(self, dim, n):
        setattr(self, 'res'+str(n)+'_1', Residual(dim, dim))
        setattr(self, 'pool'+str(n)+'_1', nn.MaxPool2d(2,2))
        setattr(self, 'res'+str(n)+'_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n-1)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self,'res'+str(n)+'_3', Residual(dim, dim))
        setattr(self,'unsample'+str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n-1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)
def count(inp,net):

    flops, params = profile(net,inputs=(inp,) )
    # print("FLOPs=", str(flops/1e6) + '{}'.format("M"))
    # print("params=", str(params/1e3) + '{}'.format("K"))
    macs, params = clever_format([flops, params], "%.4f")
    torch.cuda.synchronize()
    start_time = time.time()
    output = net(inp)
    torch.cuda.synchronize()
    end_time = time.time()
    residual_runtime_ms = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"params:{params},macs:{macs} Runtime/ms:{residual_runtime_ms} milliseconds")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 实例化 Residual 和 BSHB 模块
    device = 'cpu'
    feat= 256
    residual_module = Residual(feat, feat).to(device)
    bshb_module = HourGlassBlock(feat, 4).to(device)
    inp = torch.randn(1, feat, 32, 32).to(device)
    residual_module(inp)
    bshb_module(inp)
    # # 计算 Residual 模块的参数数量
    # print("Residual Module Parameters:")
    # count(inp,residual_module)
    # 计算 BSHB 模块的参数数量
    print("Residual Module Parameters:")
    count(inp,residual_module)
    print("BSHB Module Parameters:")
    count(inp,bshb_module)
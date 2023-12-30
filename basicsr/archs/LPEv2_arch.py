import torch
import torch.nn as nn
import torchvision
from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
from basicsr.archs.BSRN_arch import ESDB,BSConvU

class Residual(nn.Module):
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

    def __init__(self,dim=128,fb_single_face=False):
        super(LightPriorEstimationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            Residual(64, dim),
            # ResBlock(dim),
            # ResBlock(128),
        )
        self.hg_blocks = nn.Sequential(
            BSHB(dim, 3),
            # BSHB(dim, 3),
        )
        self.con_block = nn.Sequential(
            nn.Conv2d(dim, 1 if fb_single_face else 11, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1 if fb_single_face else 11),
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
# if __name__ == "__main__":
#     x = torch.randn((1, 64, 32, 32))
#     model = LightPriorEstimationNetwork(64)
#     b = model(x)
#     # print(b.shape)
#     # model = ResBlock(128)
#     summaryv2(model, (1,64,32,32))
#     # summaryv1(model,(3,32,32))
#     # print_network(model)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 实例化 Residual 和 BSHB 模块
    model = LightPriorEstimationNetwork(64)
    residual_module = Residual(64, 64)
    bshb_module = BSHB(64, 3)

    # 计算 Residual 模块的参数数量
    residual_params = count_parameters(residual_module)
    print("Residual Module Parameters:", residual_params)

    # 计算 BSHB 模块的参数数量
    bshb_params = count_parameters(bshb_module)
    print("BSHB Module Parameters:", bshb_params)

    # 创建一个示例输入
    example_input = torch.rand(1, 64, 32, 32).to('cpu')

    # 获取 Residual 和 BSHB 模块的输出
    residual_output = residual_module(example_input)
    bshb_output = bshb_module(example_input)

    # 打印输出特征维度
    print("Residual Output Shape:", residual_output.shape)
    print("BSHB Output Shape:", bshb_output.shape)
    # 使用 torchinfo 打印模块摘要
    print("\RB-BS Module Summary:")
    summaryv2(residual_module, input_size=(1, 64, 32, 32))

    print("\nBSHB Module Summary:")
    summaryv2(bshb_module, input_size=(1, 64, 32, 32))
    # 确保输入数据也在 GPU 上
    import time
    example_input = example_input.to('cuda')
    start_time = time.time()
    residual_output = residual_module(example_input)
    end_time = time.time()
    residual_runtime_ms = (end_time - start_time) * 1000  # 转换为毫秒
    print("Residual Module Runtime:", residual_runtime_ms, "milliseconds")

    # 测量 HourGlassBlock 模块的运行时间
    start_time = time.time()
    hourglass_output = bshb_module(example_input)
    end_time = time.time()
    hourglass_runtime_ms = (end_time - start_time) * 1000  # 转换为毫秒
    print("bshb_module Module Runtime:", hourglass_runtime_ms, "milliseconds")

    summaryv2(model, (1,64,32,32))
    start_time = time.time()
    hourglass_output = model(example_input)
    end_time = time.time()
    model_ms = (end_time - start_time) * 1000  # 转换为毫秒
    print("module Module Runtime:", model_ms, "milliseconds")
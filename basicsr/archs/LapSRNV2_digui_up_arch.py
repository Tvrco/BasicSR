import torch
import torch.nn as nn
import numpy as np
import math
import basicsr.archs.Upsamplers as Upsamplers

from basicsr.utils.registry import ARCH_REGISTRY

from torchinfo import summary as summaryv2
from torchsummary import summary as summaryv1
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1)     # 计算上采样因子
    if size % 2 == 1:       # 如果 size 是奇数，则中心位置为 factor - 1；如果是偶数，则中心位置为 factor - 0.5
        center = factor - 1 
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]  # 创建原点网格
     # 创建二维双线性上采样核
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()  # 返回二维双线性上采样核的张量表示

class RecursiveBlock(nn.Module):
    def __init__(self, d):
        super(RecursiveBlock, self).__init__()
        self.block = nn.Sequential()
        for i in range(d):   # 对于给定的循环次数 d，创建前向传播模块的序列
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))
            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
    def forward(self, x):   # 定义前向传播函数
        output = self.block(x)  # 应用前向传播模块的序列
        return output

class FeatureEmbedding(nn.Module):
    def __init__(self, r, d):
        super(FeatureEmbedding, self).__init__()
        self.recursive_block = RecursiveBlock(d)   # 创建递归块对象
        self.num_recursion = r    # 存储递归的次数
    def forward(self, x):   # 定义前向传播函数
        output = x.clone()   # 克隆输入张量以确保梯度不会流回输入
        for i in range(self.num_recursion):  # 执行给定数量的递归
            output = self.recursive_block(output) + x  # 将递归块应用到输出上并添加输入张量
        return output


@ARCH_REGISTRY.register()
class LapSrnMSV2(nn.Module):
    def __init__(self,
                 r=8, 
                 d=5, 
                 upscale=8,
                 num_in_ch=3,
                 num_feat=64):
        super(LapSrnMSV2, self).__init__()

        self.scale = upscale
        # 定义输入层卷积层，将输入图像变换为64通道特征图
        self.conv_input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=True)

        # 定义转置卷积层，将64通道特征图转换为64通道上采样特征图
        self.transpose = nn.ConvTranspose2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3,
                                            stride=2, padding=0, bias=True)
        self.pre_upsampler = Upsamplers.PixelShuffleDirect(scale=2, num_feat=num_feat, num_out_ch=64)
        # 定义LeakyReLU激活函数
        self.relu_features = nn.LeakyReLU(0.2, inplace=True)
        # 定义上采样层，将输入图像进行上采样
        self.scale_img = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4,
                                            stride=2, padding=0, bias=False)
        self.img_upsampler = Upsamplers.PixelShuffleDirect(scale=2, num_feat=3, num_out_ch=3)

        # 定义预测层，将64通道特征图转换为1通道输出图像
        self.predict = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

        # 定义特征嵌入层
        self.features = FeatureEmbedding(r, d)

        i_conv = 0
        i_tconv = 0

        # 遍历所有层，初始化权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 初始化卷积层权重
                if i_conv == 0:
                    m.weight.data = 0.001 * torch.randn(m.weight.shape)
                else:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)

                # 初始化卷积层偏置
                if m.bias is not None:
                    m.bias.data.zero_()

                i_conv += 1

            if isinstance(m, nn.ConvTranspose2d):
                # 初始化转置卷积层权重
                if i_tconv == 0:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)
                else:
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

                # 初始化转置卷积层偏置
                if m.bias is not None:
                    m.bias.data.zero_()

                i_tconv += 1

    # 前向传播函数
    def forward(self, x):
        # 对输入的x先进行一次卷积操作，得到features
        # print("x_shape:".ljust(15),x.shape)
        features = self.conv_input(x)
        # 初始化一个列表，用于存储输出的图片
        output_images = []
        # 克隆输入的x，用于生成不同尺度的图片
        rescaled_img = x.clone()

        # 根据需要生成不同尺度的图片
        for i in range(int(math.log2(self.scale))):
            # 使用FeatureEmbedding模块对features进行特征嵌入操作
            features = self.features(features)
            # print("features:".ljust(15),features.shape) # torch.Size([1, 64, 16, 16])
            features = self.pre_upsampler(self.relu_features(features)) # features_out: torch.Size([1, 64, 32, 32])
            # print("features_out:".ljust(15),features.shape) 
    
            # 使用卷积操作对features进行特征图回归，得到回归结果
            predict = self.predict(features)  # rescaled_img: torch.Size([1, 3, 32, 32])
            # print("predict:".ljust(15),predict.shape)

            # 使用反卷积对rescaled_img进行上采样，得到高分辨率图片
            rescaled_img = self.img_upsampler(rescaled_img) # GT的上采样 torch.Size([1, 3, 32, 32])
            # print("rescaled_img:".ljust(15),rescaled_img.shape)
            # 对rescaled_img进行裁剪，裁剪掉左右和顶部1列像素，为了使其与features的尺寸相同
            # rescaled_img = rescaled_img[:, :, 1:-1, 1:-1]


            # 将回归结果和rescaled_img相加得到最终的高分辨率图片
            out = torch.add(predict, rescaled_img) # torch.Size([1, 3, 32, 32])
            # print("out:".ljust(15),out.shape)
            # 对输出的图片进行限幅，使其像素值在0到1之间
            out = torch.clamp(out, 0.0, 1.0)
            # 将输出的图片添加到列表中
            output_images.append(out)

        # 返回输出的图片列表
        return output_images

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

    model = LapSrnMS(r=5, d=2, scale=8).cuda()
    print(model)
    count_parameters(model)
    # print(count_conv_layers(model))
    summaryv2(model, (1,3,16,16))
    # summaryv1(model,(3,16,16))
    # output_images = model(x)
    # for i, img in enumerate(output_images):
        # print(f"Layer {i+1}".ljust(5),f": {img.shape}")
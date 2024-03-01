import torch
import torch.nn as nn
from basicsr.archs import RFDN_block as B
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
def make_model(args, parent=False):
    model = RFDN()
    return model

@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=8):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
if __name__ == "__main__":
    model = RFDN(upscale=8)
    # summaryv2(model, (1,3,16,16))
    # summaryv1(model,(3,128,128))
    # print(model)

    input_data = torch.randn((1, 3, 16, 16))
    output_data = model(input_data)
    print(output_data.shape)
    macs, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {macs / 1e6}M, FLOPs: {macs / 1e9}G,Params: {params / 1e3}K")
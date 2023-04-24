import torch
import time
from basicsr.archs.rlfn_arch import RLFN


def test_rlfn():
    """Test arch: RLFN."""

    # model init and forward
    net = RLFN(in_channels=3, out_channels=3, feature_channels=48, mid_channels=48, upscale=4).cuda()
    img = torch.rand((1, 3, 16, 16), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 3, 64, 64)

    # ----------------- the x3 case ---------------------- #
    # net = RLFN(num_in_ch=1, num_out_ch=1, num_feat=4, num_block=1, upscale=2).cuda()
    # img = torch.rand((1, 1, 16, 16), dtype=torch.float32).cuda()
    # output = net(img)
    # assert output.shape == (1, 1, 48, 48)

import torch
import torch.nn as nn
from basicsr.archs.FAN import FAN
from basicsr.losses.basic_loss import L1Loss
from torch.utils.model_zoo import load_url
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class FANloss(nn.Module):
    def __init__(self,loss_weight=0.1):
        super(FANloss, self).__init__()
        self.loss_weight = loss_weight
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net
        self.criterion = nn.MSELoss()
        # self.l1loss = L1Loss(loss_weight=1.0, reduction='mean')
    def forward(self, data, target):
        # data = self.FAN_net(data)
        # target = self.FAN_net(target)
        # print(data[0].size())
        # print(target[0].size())
        # exit()
        # return self.l1loss(data, target) + self.loss_weight * self.criterion(self.FAN_net(data)[0], self.FAN_net(target.detach())[0])
        return self.loss_weight * self.criterion(self.FAN_net(data)[0], self.FAN_net(target.detach())[0])

if __name__ == "__main__":
    net = FANloss()
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    print(net(x, y))
    exit()
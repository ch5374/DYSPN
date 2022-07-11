from hyptorch.nn import *
import torch.nn as nn
import torch
import torch.nn.functional as F


class HypCNN(nn.Module):
    def __init__(self, in_channels, out_channels, oheight, owidth, config):
        super(HypCNN, self).__init__()

        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)
        self.c = 0.1
        self.sigmoid = nn.Sigmoid()
        if config['non_linearity']:
            print("ReLU!!")
            self.nonlin = nn.ReLU(inplace=True)
        else:
            self.nonlin = None
        self.hypConv = GeoConv(in_channels, out_channels, self.c, padding=1, nonlin=self.nonlin,
                               train_c=config['train_c'], mode=config['hypCNN_mode'])
        if config['batchnorm']:
            print("BatchNorm!!")
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        x_ = self.hypConv(x)
        if self.bn:
            return self.bn(self.sigmoid(x_))
        else:
            return self.sigmoid(x_)


class Unpool(nn.Module):
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride
        self.weights = torch.zeros(int(num_channels), 1, stride, stride)
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        self.weights = self.weights.to(x)
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
from utils import load_imagenet, evaluate_accuracy
from pytorchcv.model_provider import get_model as ptcv_get_model


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
max_act = []
resnet34 = ((3, 64, 64), (4, 64, 128), (6, 128, 256), (3, 256, 512))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norelu=False, stride=1, kernel=3, pad=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.norelu = norelu

        if not norelu:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.norelu:
            return self.activ(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class ResInitBlock(nn.Module):
    def __init__(self):
        super(ResInitBlock, self).__init__()
        self.conv = ConvBlock(3, 64, kernel=7, stride=2, pad=3)
        self.pool = nn.MaxPool2d(3,2,1)

    def forward(self, x):
        return self.pool(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pad=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(inchannel, outchannel, stride=stride, pad=pad)
        self.conv2 = ConvBlock(outchannel, outchannel, True)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Scale():
    def __init__(self):
        self.scale = 1.0

    def __call__(self, x):
        return self.scale * x


class ResUnit(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pad=1):
        super(ResUnit, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.body = ResBlock(inchannel, outchannel, stride=stride, pad=1)

        if self.inchannel == self.outchannel:
            self.identity_conv = Scale()
        else:
            self.identity_conv = ConvBlock(inchannel, outchannel, kernel=1, stride=2, pad=pad, norelu=True)
        self.activ = nn.ReLU(True)

    def forward(self, x):
        return self.activ(self.body(x)+self.identity_conv(x))


class CNN(nn.Module):
    def __init__(self, arch=resnet34):
        super(CNN, self).__init__()
        self.hooks = []
        features = nn.Sequential()
        features.add_module('init_block', ResInitBlock())

        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = nn.Sequential()
            for j in range(num_residual):
                if j == 0 and i!=0:
                    blk.add_module('unit%d'%(j+1), ResUnit(in_channel, out_channel, stride=2, pad=0))
                elif j==0 and i == 0:
                    blk.add_module('unit%d' % (j + 1), ResUnit(in_channel, out_channel, stride=1, pad=0))
                else:
                    blk.add_module('unit%d'%(j+1), ResUnit(out_channel, out_channel))
            features.add_module('stage%d'%(i+1), blk)

        self.features = features
        self.finalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.output = nn.Linear(512, 1000)

    def forward(self, X):
        X = self.features(X)
        X = self.finalpool(X)
        out = self.output(X.view(X.shape[0], -1))
        return out


if __name__ == '__main__':
    train_iter, test_iter, _, _ = load_imagenet(root='./data/ImageNet',batch_size=64)

    # net = CNN()
    net = ptcv_get_model('resnet34', pretrained=False)
    print(net)
    # [net.hooks[i].remove() for i in range(len(net.hooks))]
    dict = ptcv_get_model('resnet34', pretrained=True).to(device).state_dict()
    net.load_state_dict(dict)
    net.eval()

    # net = nn.DataParallel(net, device_ids = [0, 1, 2, 3])
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
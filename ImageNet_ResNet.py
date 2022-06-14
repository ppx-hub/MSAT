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


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
max_act = []
resnet34 = ((3, 64, 64), (4, 64, 128), (6, 128, 256), (3, 256, 512))


def hook(module, input, output):
    sort, _ = torch.sort(output.detach().view(-1).cpu())
    max_act.append(sort[int(sort.shape[0] * 0.99) - 1])


class BasicBlock(nn.Module):
    def __init__(self, hooks, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        stride = 2 if in_channel!=out_channel else 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)

        hooks.append(self.conv1.register_forward_hook(hook))
        hooks.append(self.bn1.register_forward_hook(hook))
        hooks.append(self.relu.register_forward_hook(hook))
        hooks.append(self.conv2.register_forward_hook(hook))
        hooks.append(self.bn2.register_forward_hook(hook))

        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, 2, bias=False),
                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1))
            hooks.append(self.downsample[0].register_forward_hook(hook))
            hooks.append(self.downsample[1].register_forward_hook(hook))

        self.relu2 = nn.ReLU(inplace=True)
        hooks.append(self.relu2.register_forward_hook(hook))

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.in_channel != self.out_channel:
            X = self.downsample(X)
        return self.relu2(X + Y)


class CNN(nn.Module):
    def __init__(self, arch=resnet34):
        super(CNN, self).__init__()
        hooks = []
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        hooks.append(self.conv1.register_forward_hook(hook))
        hooks.append(self.bn1.register_forward_hook(hook))
        hooks.append(self.relu.register_forward_hook(hook))
        hooks.append(self.maxpool.register_forward_hook(hook))

        layers = []
        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(BasicBlock(hooks, in_channel, out_channel))
                else:
                    blk.append(BasicBlock(hooks, out_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

        hooks.append(self.avgpool.register_forward_hook(hook))
        hooks.append(self.fc.register_forward_hook(hook))
        self.hooks = hooks

    def forward(self, X):
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.maxpool(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.avgpool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


if __name__ == '__main__':
    batch_size = 128
    train_iter, test_iter, _, _ = load_imagenet(root='./data/ImageNet',batch_size=batch_size)

    net = CNN()
    [net.hooks[i].remove() for i in range(len(net.hooks))]
    dict = models.resnet34(True).to(device).state_dict()
    net.load_state_dict(dict)
    net.eval()

    # net = nn.DataParallel(net, device_ids = [0, 1, 2, 3])
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
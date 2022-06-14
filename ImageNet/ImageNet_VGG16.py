import torch
import torch.nn as nn
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
from pytorchcv.model_provider import get_model as ptcv_get_model


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
max_act = []
vgg16_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))


def load_imagenet(root='/data/dataset/ILSVRC2012', batch_size=128):
    '''
    load imagenet 2012
    we use images in train/ for training, and use images in val/ for testing
    https://github.com/pytorch/examples/tree/master/imagenet
    '''
    IMAGENET_PATH = root
    traindir = os.path.join(IMAGENET_PATH, 'train')
    valdir = os.path.join(IMAGENET_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))


class VGGDense(nn.Module):
    def __init__(self, in_features, out_features):
        super(VGGDense, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.activ = nn.ReLU(True)
        self.dropout = nn.Dropout(inplace=False)

    def forward(self, x):
        return self.dropout(self.activ(self.fc(x)))


class VGGOutputBlock(nn.Module):
    def __init__(self):
        super(VGGOutputBlock, self).__init__()

        self.fc1 = VGGDense(25088, 4096)
        self.fc2 = VGGDense(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class CNN(nn.Module):
    def __init__(self, arch=vgg16_arch):
        super(CNN, self).__init__()
        self.hooks = []
        blks = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(arch):
            kkk = nn.Sequential()
            for j in range(num_convs):
                if j==0:
                    kkk.add_module('unit{:d}'.format(j+1), ConvBlock(in_channels, out_channels))
                else:
                    kkk.add_module('unit{:d}'.format(j + 1), ConvBlock(out_channels, out_channels))
            kkk.add_module('pool{:d}'.format(i+1), nn.MaxPool2d(2, 2))
            blks.add_module('stage{:d}'.format(i+1), kkk)

        self.features = blks
        self.output = VGGOutputBlock()

    def forward(self, X):
        feature = self.features(X)
        output = self.output(feature.view(feature.shape[0], -1))
        return output


def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in tqdm(data_iter):
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
    return acc_sum / n


if __name__ == '__main__':
    train_iter, test_iter, _, _ = load_imagenet(root='../data/ImageNet', batch_size=100)

    net = ptcv_get_model('bn_vgg16', pretrained=True).to(device)
    print(net)
    net.eval()

    # net = nn.DataParallel(net, device_ids = [0, 1, 2, 3])
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
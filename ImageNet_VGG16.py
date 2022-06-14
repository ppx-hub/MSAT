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


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
max_act = []
vgg16_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))


def load_imagenet(root='/data/raid/floyed/ILSVRC2012', batch_size=128):
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


def hook(module, input, output):
    sort, _ = torch.sort(output.detach().view(-1).cpu())
    max_act.append(sort[int(sort.shape[0] * 0.9995) - 1])
    print('aaa')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*blk)


class CNN(nn.Module):
    def __init__(self, arch=vgg16_arch):
        super(CNN, self).__init__()
        hooks = []
        blks = []
        for i, (num_convs, in_channels, out_channels) in enumerate(arch):
            block = vgg_block(num_convs, in_channels, out_channels)
            for j in range(num_convs * 3 + 1):
                blks.append(block[j])

        self.features = nn.Sequential(*blks)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000))

        for i in range(len(self.features)):
            hooks.append(self.features[i].register_forward_hook(hook))
        hooks.append(self.avgpool.register_forward_hook(hook))
        for i in range(len(self.classifier)):
            hooks.append(self.classifier[i].register_forward_hook(hook))
        self.hooks = hooks

    def forward(self, X):
        feature = self.features(X)
        feature = self.avgpool(feature)
        output = self.classifier(feature.view(feature.shape[0], -1))
        return output

def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
    return acc_sum / n


if __name__ == '__main__':
    train_iter, test_iter, _, _ = load_imagenet(root='./data/ImageNet',batch_size=100)

    net = CNN()
    [net.hooks[i].remove() for i in range(len(net.hooks))]
    dict = models.vgg16_bn(True).state_dict()
    net.load_state_dict(dict)
    net.eval()

    # net = nn.DataParallel(net, device_ids = [0, 1, 2, 3])
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
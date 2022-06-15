import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import random
from utils import *

# device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
resnet18 = ((2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512))

def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        stride = 2 if in_channel!=out_channel else 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)

        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, 2, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1))

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.in_channel != self.out_channel:
            X = self.downsample(X)
        return self.relu2(X + Y)


class ResNet20(nn.Module):
    def __init__(self, arch=resnet18):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1), nn.ReLU(inplace=True))

        layers = []
        for i, (num_residual, in_channel, out_channel) in enumerate(arch):
            blk = []
            for j in range(num_residual):
                if j == 0:
                    blk.append(BasicBlock(in_channel, out_channel))
                else:
                    blk.append(BasicBlock(out_channel, out_channel))
            layers.append(nn.Sequential(*blk))

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 100)

    def forward(self, X):
        X = self.conv1(X)
        X = self.layer4(self.layer3(self.layer2(self.layer1(X))))
        X = self.pool(X)
        out = self.fc(X.view(X.shape[0], -1))
        return out


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='mse'):
    net = net.to(device)
    best = 0
    print("training on ", device)
    if losstype == 'mse':
       loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            label = y
            if losstype == 'mse':
                label = F.one_hot(y, 100).float()
            l = loss(y_hat, label)
            losss.append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        # # plt.figure(1)
        # plt.plot(losses)
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # # plt.show()
        # plt.savefig('./loss_curve/CIFAR100_ResNet20.jpg')

        if test_acc > best:
            torch.save(net.state_dict(), 'saved_model/CIFAR100_ResNet20.pth')
            best = test_acc


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
    seed_all(356)
    batch_size = 128
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          CIFAR10Policy(),
                                          transforms.ToTensor(),
                                          Cutout(n_holes=1, length=16),
                                          normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    cifar100_train = datasets.CIFAR100(root='../data/', train=True, download=False, transform=transform_train)
    cifar100_test = datasets.CIFAR100(root='../data/', train=False, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4)

    lr, num_epochs = 0.1, 300
    net = ResNet20()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    # train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='crossentry')

    net.load_state_dict(torch.load("../saved_model/CIFAR100_ResNet20.pth"))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)

from turtle import forward
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
from tqdm import tqdm
import random
from utils import *
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VGG16(nn.Module):
    def __init__(self, relu_max=1):  # 1   3e38
        super(VGG16, self).__init__()
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.conv = cnn
        self.fc = nn.Linear(512, 10, bias=True)

    def forward(self, input, compute_efficiency=None):
        if compute_efficiency is True:
            x = input
            all = 1769472 + 37748736 + 18874368 + 37748736 + 18874368 + 37748736 + 37748736 + \
                  18874368 + 37748736 + 37748736 + 9437184 + 9437184 + 9437184 + 5120
            firing_rate = [
                0.045486677438020706, 0.07766489684581757, 0.05115384981036186,
                0.04537772759795189, 0.04539765790104866, 0.03777753934264183,
                0.016239548102021217, 0.026328597217798233, 0.010158049874007702,
                0.017707584425807, 0.05968906730413437, 0.06104709580540657,
                0.053790975362062454
            ] # len is 13
            snn_op = 0
            index = 1
            for name, layer in self.conv.named_modules():
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                    print("relu{}: {}".format(index, x.shape[2] * x.shape[3] * layer.kernel_size[0] * layer.kernel_size[1]
                                              * layer.in_channels * layer.out_channels / all))
                    if layer.in_channels != 3:
                        snn_op += x.shape[2] * x.shape[3] * layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels * layer.out_channels / all * firing_rate[index - 2]
                        print("use index:{}".format(index -2))
                    index += 1
                elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d):
                    x = layer(x)
            x = x.view(x.shape[0], -1)
            output = self.fc(x)
            print("last op:{}".format(self.fc.in_features * self.fc.out_features / all))
            snn_op += self.fc.in_features * self.fc.out_features / all * firing_rate[12]
        else:
            conv = self.conv(input)
            x = conv.view(conv.shape[0], -1)
            output = self.fc(x)
        return output


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='mse'):
    best = 0
    net = net.to(device)
    print("training on ", device)
    if losstype == 'mse':
       loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
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
                label = F.one_hot(y, 10).float()
            l = loss(y_hat, label)
            losss.append(l.cpu().item())
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

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), 'saved_model/CIFAR100_VGG16.pth')
            # print('model has been saved!')


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
    seed_all(42)
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
    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4,  pin_memory=True)
    print('dataloader finished')

    lr, num_epochs = 0.1, 300
    net = VGG16()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    # train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='crossentropy')

    net.load_state_dict(torch.load("../saved_model/CIFAR100_VGG16.pth", map_location=device))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
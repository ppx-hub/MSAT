import sys
sys.path.append('..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import matplotlib
# matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
#import matplotlib.pyplot as plt
import time
import os
from CIFAR10_VGG16 import VGG16
from CIFAR10_ResNet import ResNet20
import argparse
#from audtorch.metrics.functional import pearsonr
from utils import Converter, clean_mem_spike, evaluate_accuracy, Scale, SNode


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=0.995, type=float, help='percentile for data normalization. 0-1')  # 0.986
parser.add_argument('--spi', default=256, type=int, help='spi time')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--spicalib', default=False, type=bool, help='use spike calibration')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--monitor', default=False, type=bool, help='record inter states')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--print_pcc', default=False, type=bool, help='print pearson and theta')
parser.add_argument('--print_sin', default=False, type=bool, help='print SIN')
parser.add_argument('--plot_fsa', default=False, type=bool, help='plot the relation between first spike and activation.')
parser.add_argument('--plot_mem', default=False, type=bool, help='plot the relation between mem and activation.')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name', choices=['vgg16', 'resnet20'])
parser.add_argument('--train_batch', default=512, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=1000, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def evaluate_snn(test_iter, snn, net, device=None, duration=50, monitor=False, print_pcc=False, print_sin=False, plot_fsa=False, plot_mem=False):
    accs = []
    acc_sum, n = 0.0, 0
    snn.eval()
    spike_rate_dict = {
        'relu1': [[] for x in range(8)],
        'relu2': [[] for x in range(8)],
        'relu3': [[] for x in range(8)],
        'relu4': [[] for x in range(8)],
        'relu5': [[] for x in range(8)],
        'relu6': [[] for x in range(8)],
        'relu7': [[] for x in range(8)],
        'relu8': [[] for x in range(8)],
        'relu9': [[] for x in range(8)],
        'relu10': [[] for x in range(8)],
        'relu11': [[] for x in range(8)],
        'relu12': [[] for x in range(8)],
        'relu13': [[] for x in range(8)]
    }
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        out = 0
        folder_path = "/data1/hexiang/newframework/conversion3/CIFAR10/result/"
        if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)
        with torch.no_grad():
            clean_mem_spike(snn)
            index = 1
            for t in range(duration):
                out += snn(test_x)
                if (t + 1) % 32 == 0:
                    for name, layer in snn.named_modules():
                        if isinstance(layer, nn.Sequential) and len(layer) == 3 and isinstance(layer[0], Scale):
                            spike_rate_dict['relu' + str(index)][(t + 1) // 32 - 1].append(
                                (layer[1].all_spike.sum() / layer[1].all_spike.view(-1).shape[0]).cpu())
                            index += 1
                    index = 1
        if ind >= 1:
            break
    f = open('{}/result_usebatch{}.txt'.format(folder_path, 309), 'w')
    for x in range(8):
        index = 1
        f.write("-----------------timestep:{}-----------------\n".format((x + 1) * 32))
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Sequential) and len(layer) == 3 and isinstance(layer[0], Scale):
                f.write("relu{}: average spike number: {}\n".format(index, torch.stack(
                    spike_rate_dict['relu' + str(index)][x]).mean()))
                index += 1
    for x in range(8):
        index = 1
        f.write("-----------------timestep:{}-----------------\n".format((x + 1) * 32))
        for name, layer in net.named_modules():
            if isinstance(layer, SNode):
                f.write("relu{}: average spike rate: {}\n".format(index, torch.stack(
                    spike_rate_dict['relu' + str(index)][x]).mean() / ((x + 1) * 32)))
                index += 1


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    normalize = torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    cifar10_train = datasets.CIFAR10(root='../data/', train=True, download=False, transform=transform_train)
    cifar10_test = datasets.CIFAR10(root='../data/', train=False, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=False, num_workers=0)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model_name == 'vgg16':
        net = VGG16()
        net.load_state_dict(torch.load("../saved_model/CIFAR10_VGG16.pth", map_location=device))
    elif args.model_name == 'resnet20':
        net = ResNet20()
        net.load_state_dict(torch.load("../saved_model/CIFAR10_ResNet20.pth", map_location=device))
    else:
        raise NameError

    net.eval()
    net = net.to(device)
    # print(net)
    data = iter(test_iter).next()[0].to(device)
    data = data[0, :, :, :].unsqueeze(0)
    net(data, compute_efficiency=True)  # 这里的参数False没有用，需要调
    acc = evaluate_accuracy(test_iter, net, device)
    print("acc on ann is : {:.4f}".format(acc))

    net1 = deepcopy(net)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    cifar10_train = datasets.CIFAR10(root='../data/', train=True, download=False, transform=transform_train)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=False, num_workers=0)

    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, True, allowance=args.T//2)
    snn = converter(net)
    # snn = fuseConvBN(snn)


    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=False, num_workers=0)
    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, False, allowance=args.T//2)
    net1 = converter(net1)
    # net1 = fuseConvBN(net1)

    evaluate_snn(test_iter, snn, net1, device, args.T, args.monitor, args.print_pcc, args.print_sin, args.plot_fsa, args.plot_mem)

import sys
sys.path.append('..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
from CIFAR10_VGG16 import VGG16
from CIFAR10_ResNet import ResNet20
import argparse
from audtorch.metrics.functional import pearsonr
from utils import Converter, clean_mem_spike, seed_all
from bayes_opt import BayesianOptimization


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=0.995, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--spicalib', default=False, type=bool, help='use spike calibration')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--lateral_inhi', default=False, type=bool, help='LIPooling')
parser.add_argument('--monitor', default=False, type=bool, help='record inter states')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--print_pcc', default=True, type=bool, help='print pearson and theta')
parser.add_argument('--print_sin', default=True, type=bool, help='print SIN')
parser.add_argument('--plot_fsa', default=False, type=bool, help='plot the relation between first spike and activation.')
parser.add_argument('--plot_mem', default=True, type=bool, help='plot the relation between mem and activation.')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name', choices=['vgg16', 'resnet20'])
parser.add_argument('--train_batch', default=256, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=1000, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
args = parser.parse_args()

print("Setting Arguments.. : ", args)
print("----------------------------------------------------------")
device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def evaluate_snn(test_iter, snn, net, device=None, duration=50, monitor=False, print_pcc=False, print_sin=False, plot_fsa=False, plot_mem=False):
    accs = []
    acc_sum, n = 0.0, 0
    snn.eval()

    rout = []
    aout = []
    for ind, (test_x, test_y) in enumerate(test_iter):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        # print(test_x[0,0])
        n = test_y.shape[0]
        out = 0
        r_out = []
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            for _ in tqdm(enumerate(range(duration))):
            # for _ in enumerate(range(duration)):
                start = time.time()
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
                r_out.append(out.cpu().detach())
            r_out = (torch.stack(r_out, 0).permute(1, 2, 0).cpu().detach())[:,:,-1] / duration
            rout.append(r_out)
            aout.append(net(test_x).cpu().detach())
        break
    rout = torch.cat(rout, 0).cpu().detach()
    aout = torch.cat(aout, 0).cpu().detach()
    kl = F.kl_div(rout.softmax(dim=-1).log(), aout.softmax(dim=-1), reduction='sum')
    # print("KL_div_sum: %.6f" % kl)

    return kl


def black_box_function(p):
    p = p
    normalize = torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

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
    net1 = deepcopy(net)

    # seed_all(args.seed)
    cifar10_train = datasets.CIFAR10(root='../data/', train=True, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=True, num_workers=0)
    converter = Converter(train_iter, device, p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, True, allowance=args.T//2)
    snn = converter(net)
    # snn = fuseConvBN(snn)

    # seed_all(args.seed)
    # cifar100_train = datasets.CIFAR100(root='../data/', train=True, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=True, num_workers=0)
    converter = Converter(train_iter, device, p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, False, allowance=args.T//2)
    net1 = converter(net1)
    # net1 = fuseConvBN(net1)

    # seed_all(args.seed)
    # cifar100_train = datasets.CIFAR100(root='../data/', train=True, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    kl = evaluate_snn(train_iter, snn, net1, device, args.T, args.monitor, args.print_pcc, args.print_sin, args.plot_fsa, args.plot_mem)
    return -kl


pbounds = {'p': (0.95, 1.00)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1
)

optimizer.maximize(init_points=5, n_iter=30)
print(optimizer.max)
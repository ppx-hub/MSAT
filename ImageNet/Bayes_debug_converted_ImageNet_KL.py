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
import argparse
from audtorch.metrics.functional import pearsonr
from utils import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from bayes_opt import BayesianOptimization


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--spi', default=256, type=int, help='spi time')
parser.add_argument('--p', default=0.999, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--spicalib', default=False, type=bool, help='use spike calibration')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--lateral_inhi', default=False, type=bool, help='LIPooling')
parser.add_argument('--monitor', default=False, type=bool, help='record inter states')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--device', default='1', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--train_batch', default=100, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=50, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
args = parser.parse_args()

print("Setting Arguments.. : ", args)
print("----------------------------------------------------------")
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda:0") if args.cuda else 'cpu'
seed_all(seed=args.seed)


def evaluate_snn(test_iter, snn, net, device=None, duration=50, plot=False, linetype=None):
    snn.eval()
    rout = []
    aout = []
    for _, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        r_out = []
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            # for t in tqdm(range(duration)):
            for t in range(duration):
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
                r_out.append(out.cpu().detach())
            r_out = (torch.stack(r_out, 0).permute(1, 2, 0).cpu().detach())[:, :, -1] /duration
            rout.append(r_out)
            aout.append(net(test_x).cpu().detach())
        break
        accs.append(np.array(acc))
    rout = torch.cat(rout, 0).cpu().detach()
    aout = torch.cat(aout, 0).cpu().detach()
    kl = F.kl_div(rout.softmax(dim=-1).log(), aout.softmax(dim=-1), reduction='sum')
    return kl


def black_box_function(p):
    p = p
    train_iter, test_iter, _, _ = load_imagenet(root='../data/ImageNet', batch_size=args.batch_size, train_batch=args.train_batch)

    if args.model_name == 'vgg16':
        net = ptcv_get_model('bn_vgg16', pretrained=True).to(device)
    elif args.model_name == 'resnet34':
        net = ptcv_get_model('resnet34', pretrained=True).to(device)
    else:
        raise NameError

    net.eval()
    net = net.to(device)

    # acc = evaluate_accuracy(test_iter, net, device)
    # print("acc on ann is : {:.4f}".format(acc))

    converter = Converter(train_iter, device, p, args.channelnorm, args.lateral_inhi,
                          args.gamma, args.spicalib, args.monitor, args.smode, args.spi)
    snn = converter(net)
    torch.cuda.empty_cache()

    kl = evaluate_snn(test_iter, snn, net, device, duration=args.T)
    return -kl

pbounds = {'p': (0.99, 0.9999)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1
)

optimizer.maximize(init_points=5, n_iter=30)
print(optimizer.max)
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
from utils import Converter, clean_mem_spike, evaluate_accuracy


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=1, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--device', default='2', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--train_batch', default=512, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
parser.add_argument('--VthHand', default=1, type=float, help='Vth scale, -1 means variable')
parser.add_argument('--useDET', action='store_true', default=False, help='use DET')
args = parser.parse_args()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def evaluate_snn(test_iter, snn, net, device=None, duration=50, monitor=False, print_pcc=False, print_sin=False, plot_fsa=False, plot_mem=False):
    folder_path = "./result_conversion_{}/snn_timestep{}_p{}_LIPooling{}_Burst{}".format(
            args.model_name, duration, args.p, args.lateral_inhi, args.gamma)
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    snn.eval()
    rout = []
    aout = []
    accs = []
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        r_out = []
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            for t in range(duration):
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
                r_out.append(out.cpu().detach())
            r_out = (torch.stack(r_out, 0).permute(1, 2, 0).cpu().detach())[:,:,-1] / duration
            rout.append(r_out)
            aout.append(net(test_x).cpu().detach())
        # break
        accs.append(np.array(acc))
    rout = torch.cat(rout, 0).cpu().detach()
    aout = torch.cat(aout, 0).cpu().detach()
    kl = F.kl_div(rout.softmax(dim=-1).log(), aout.softmax(dim=-1), reduction='sum')
    if True:
        f = open('{}/result.txt'.format(folder_path), 'w')
        f.write("Setting Arguments.. : {}\n".format(args))
        accs = np.array(accs).mean(axis=0)
        for iii in range(256):
            if iii == 0 or iii == 3 or iii == 7 or (iii + 1) % 16 == 0:
                f.write("timestep {}:{}\n".format(str(iii+1).zfill(3), accs[iii]))
        f.write("max accs: {}, timestep:{}\n".format(max(accs), np.where(accs == max(accs))))
        f.write("KL_div_sum: {:.6f}\n".format(kl))
        f.write("use for latex tabular: & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.6f}".format(max(accs) * 100, accs[31]* 100,
                                                                                         accs[63]*100, accs[127]*100, accs[255]*100,kl))
        f.close()
        accs = torch.from_numpy(accs)
        torch.save(accs, "{}/accs.pth".format(folder_path))

if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    cifar10_train = datasets.CIFAR10(root='../data/', train=True, download=False, transform=transform_train)
    cifar10_test = datasets.CIFAR10(root='../data/', train=False, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=args.train_batch, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    acc = evaluate_accuracy(test_iter, net, device)
    print("acc on ann is : {:.4f}".format(acc))

    net1 = deepcopy(net)


    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, True, allowance=args.spi)
    snn = converter(net)
    # snn = fuseConvBN(snn)


    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, False, allowance=args.spi)
    net1 = converter(net1)
    # net1 = fuseConvBN(net1)

    evaluate_snn(test_iter, snn, net1, device, args.T, args.monitor, args.print_pcc, args.print_sin, args.plot_fsa, args.plot_mem)

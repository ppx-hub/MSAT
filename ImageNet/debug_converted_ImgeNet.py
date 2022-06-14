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


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=0.9999, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--spicalib', default=False, type=bool, help='use spike calibration')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--monitor', default=True, type=bool, help='record inter states')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--print_pcc', default=True, type=bool, help='print pearson and theta')
parser.add_argument('--print_sin', default=True, type=bool, help='print SIN')
parser.add_argument('--plot_fsa', default=False, type=bool, help='plot the relation between first spike and activation.')
parser.add_argument('--plot_mem', default=True, type=bool, help='plot the relation between mem and activation.')
parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name', choices=['vgg16', 'resnet20'])
parser.add_argument('--train_batch', default=30, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()


def evaluate_snn(test_iter, snn, net, device=None, duration=50, monitor=False, print_pcc=False, print_sin=False, plot_fsa=False, plot_mem=False):
    accs = []
    acc_sum, n = 0.0, 0
    snn.eval()

    conv_indexs = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]    # len(conv_indexs)=13
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    aconv, rmem, rspike = [], [], []

    for ind, (test_x, test_y) in enumerate(test_iter):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        rout = []
        out = 0
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            # for t in tqdm(range(duration)):
            for _ in tqdm(enumerate(range(duration))):
                start = time.time()
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
                if monitor: rout.append(out.cpu().detach())

            if monitor:
                aout = net(test_x).cpu().detach()
                pbar = tqdm(list(np.arange(len(conv_indexs))))
                for iii in range(len(conv_indexs)):
                    aconv.append(net.conv[conv_indexs[iii]][1].sum.cpu().detach())
                    rmem.append(torch.stack(snn.conv[conv_indexs[iii]][1].rmem, 0).permute(1, 2, 3, 4, 0).cpu().detach())
                    rspike.append(torch.stack(snn.conv[conv_indexs[iii]][1].rspike, 0).permute(1, 2, 3, 4, 0).cpu().detach())
                    pbar.update(1)
                    pbar.set_description('getting the conv_%s values ...' % conv_indexs[iii])
                pbar.close()
                aconv.append(aout)
                rmem.append(torch.stack(rout, 0).permute(1, 2, 0).cpu().detach())

                if print_pcc:
                    print('-' * 20, 'print pcc', '-' * 20)
                    for iii in range(len(conv_indexs)):
                        a = aconv[iii].view(aconv[iii].shape[0], -1)
                        b = rmem[iii][:,:,:,:,-1].view(aconv[iii].shape[0], -1) / duration
                        pcc = pearsonr(a, b)
                        theta = torch.acos(F.cosine_similarity(a, b))
                        print('Conv_%s: pcc_avg: %.4f, pcc_std: %.4f, theta_avg: %.4f, theta_std: %.4f' % (conv_indexs[iii], pcc.mean(), pcc.std(), theta.mean(), theta.std()))
                
                if print_sin:
                    print('-' * 20, 'print sin radio and degree', '-' * 20)
                    for iii in range(len(conv_indexs)):
                        sin_ra = (rspike[iii][aconv[iii] < 0][:, -1] > 1).sum() / (aconv[iii] < 0).sum()
                        if sin_ra != 0:
                            sin_de =  (rspike[iii][aconv[iii] < 0][:, -1]).sum() / (rspike[iii][aconv[iii] < 0][:, -1] > 1).sum() / duration
                        else:
                            sin_de = 0
                        print('Conv_%s: sin_radio: %.4f, sin_degree: %.4f' % (conv_indexs[iii], sin_ra, sin_de))

                if plot_fsa:
                    fsa_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    plt.figure(figsize=(18, 24))
                    for iii, index in enumerate(fsa_list):
                        aconvx = aconv[index]
                        rspikex = rspike[index]
                        ax = plt.subplot(3, 3, iii+1)
                        a = aconvx[aconvx>0]
                        b = rspikex[aconvx>0].argmax(1)
                        plt.scatter(a, b)
                        plt.plot((np.arange(duration) + 1) / duration, [np.floor(1 / (i + 1) * duration) for i in range(duration)], linewidth=1, color='r')
                        plt.xlim([0,1.2])
                        plt.ylim([0,60])
                        ax.set_title("Conv%s" % conv_indexs[index])
                    print('finish plotting relationship of the time of first spike and activation!\n saving...')
                    plt.savefig('./result/result_first_spike_time.jpg')

                if plot_mem:
                    print('-' * 20, 'plot totalmem', '-' * 20)
                    mem_list = [2, 3, 4, 6, 8, 10, 11, 12]
                    plt.figure(figsize=(18, 24))
                    for iii, index in enumerate(mem_list):
                        ax = plt.subplot(3, 3, iii+1)
                        for i in range(9):
                            plt.plot(np.arange(duration), rmem[index][1, i+11, 0, 0].numpy(), color=color[i], linewidth=2)
                            plt.plot(np.arange(duration), np.arange(duration) * aconv[index][1, i+11, 0, 0].numpy(), color=color[i], linewidth=1)
                        ax.set_title("Conv%s" % conv_indexs[index])
                    
                    ax = plt.subplot(339)
                    for i in range(8):
                        plt.plot(np.arange(duration), rmem[-1][2, i].numpy(), color=color[i], linewidth=2)
                        plt.plot(np.arange(duration), np.arange(duration) * aconv[-1][2, i].numpy(), color=color[i], linewidth=1)
                    ax.set_title("OUT")
                    plt.savefig('../result/imagenet/result_ann_snn.pdf', dpi=600)

                # # 测试max错误占比
                # opration = nn.MaxPool2d(kernel_size=2, stride=2)
                # max_ind = [6, 13, 23, 33, 43]
                # m = []
                # for iii in range(len(max_ind)):
                #     real = opration(snn.conv[max_ind[iii]].input) / duration
                #     a = snn.conv[max_ind[iii]].sum[snn.conv[max_ind[iii]].sum > 0]/duration
                #     b = real[snn.conv[max_ind[iii]].sum > 0]
                #     m.append(((a-b).abs() > 2/duration).sum() / b.numel())
                # m = [1-m[i] for i in range(len(m))]
            break

    if not monitor:
        accs = np.array(accs).mean(axis=0)
        for iii in [3, 7, 15, 31, 63, 127]:
            print("timestep", str(iii+1).zfill(3) + ':', accs[iii])
        print(max(accs), np.where(accs == max(accs)))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    seed_all(seed=args.seed)
    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    train_iter, test_iter, _, _ = load_imagenet(root='../data/ImageNet', batch_size=args.batch_size, train_batch=args.train_batch)

    if args.model_name == 'vgg16':
        net = ptcv_get_model('bn_vgg16', pretrained=True).to(device)
    elif args.model_name == 'resnet34':
        net = ptcv_get_model('resnet34', pretrained=True).to(device)
    else:
        raise NameError

    net.eval()
    net = net.to(device)

    net1 = deepcopy(net)
    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, True)
    snn = converter(fuseConvBN(net))

    converter = Converter(train_iter, device, args.p, args.channelnorm, args.lateral_inhi, args.gamma, args.spicalib, args.monitor, False)
    net1 = converter(fuseConvBN(net1))

    evaluate_snn(test_iter, snn, net1, device, args.T, args.monitor, args.print_pcc, args.print_sin, args.plot_fsa, args.plot_mem)

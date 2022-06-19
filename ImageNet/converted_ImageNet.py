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
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "8"  # 设置MKL-DNN CPU加速库的线程数。
import argparse
from utils import *
from pytorchcv.model_provider import get_model as ptcv_get_model


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
parser.add_argument('--train_batch', default=100, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=50, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
parser.add_argument('--VthHand', default=1, type=float, help='Vth scale, -1 means variable')
parser.add_argument('--useDET', action='store_true', default=False, help='use DET')
parser.add_argument('--useDTT', action='store_true', default=False, help='use DTT')
args = parser.parse_args()


def evaluate_snn(test_iter, snn, device=None, duration=50, plot=False, linetype=None):
    folder_path = "./result_conversion_{}/snn_p{}_VthHand{}_useDET_{}_useDTT_{}".format(
            args.model_name, args.p, args.VthHand, args.useDET, args.useDTT)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    FolderPath.folder_path = folder_path
    linetype = '-' if linetype == None else linetype
    accs = []
    snn.eval()
    rout = []
    aout = []
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
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
        accs.append(np.array(acc))
        # if ind==15:
        #     break

    if True:
        f = open('{}/result.txt'.format(folder_path), 'w')
        f.write("Setting Arguments.. : {}\n".format(args))
        accs = np.array(accs).mean(axis=0)
        for iii in range(256):
            if iii == 0 or iii == 3 or iii == 7 or (iii + 1) % 16 == 0:
                f.write("timestep {}:{}\n".format(str(iii+1).zfill(3), accs[iii]))
        f.write("max accs: {}, timestep:{}\n".format(max(accs), np.where(accs == max(accs))))
        f.write("use for latex tabular: & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(max(accs) * 100, accs[31]* 100,
                                                                                         accs[63]*100, accs[127]*100, accs[255]*100))
        f.close()
        accs = torch.from_numpy(accs)
        torch.save(accs, "{}/accs.pth".format(folder_path))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("-" * 40)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    seed_all(seed=args.seed)
    device = torch.device("cuda:0") if args.cuda else 'cpu'
    train_iter, test_iter, _, _ = load_imagenet(root='../data/ImageNet', batch_size=args.batch_size, train_batch=args.train_batch)

    if args.model_name == 'vgg16':
        net = ptcv_get_model('bn_vgg16', pretrained=True).to(device)
    elif args.model_name == 'resnet34':
        net = ptcv_get_model('resnet34', pretrained=True).to(device)
    else:
        raise NameError

    net.eval()
    net = net.to(device)

    acc = evaluate_accuracy(test_iter, net, device)
    print("acc on ann is : {:.4f}".format(acc))

    converter = Converter(train_iter, device, args.p, args.lateral_inhi,
                          args.gamma, args.smode, args.VthHand, args.useDET, args.useDTT)
    snn = converter(net)  # use threshold balancing or not
    torch.cuda.empty_cache()

    # sys.exit()
    evaluate_snn(test_iter, snn, device, duration=args.T)


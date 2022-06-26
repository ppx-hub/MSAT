# plot for VGG-cifar100
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

model_name = "resnet20"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_{}/parameters_group1/".format(model_name)
if model_name == "vgg16":
    acc_target = 0.7849
if model_name == "resnet20":
    acc_target = 0.8069
acc_list_target = [acc_target] * 256

Path_0point5 = root + 'snn_VthHand0.5_useDET_False_useDTT_False_useSC_False/accs.pth'
acc_list = torch.load(Path_0point5)
acc_list1 = [acc_list[i].item() for i in range(len(acc_list))]

Path_0point7 = root + 'snn_VthHand0.7_useDET_False_useDTT_False_useSC_False/accs.pth'
acc_list = torch.load(Path_0point7)
acc_list2 = [acc_list[i].item() for i in range(len(acc_list))]

Path_0point9 = root + 'snn_VthHand0.9_useDET_False_useDTT_False_useSC_False/accs.pth'
acc_list = torch.load(Path_0point9)
acc_list3 = [acc_list[i].item() for i in range(len(acc_list))]

Path_With_DTT = root + 'snn_VthHand-1.0_useDET_False_useDTT_True_useSC_False/accs.pth'
acc_list = torch.load(Path_With_DTT)
acc_list4 = [acc_list[i].item() for i in range(len(acc_list))]

Path_With_DET = root + 'snn_VthHand-1.0_useDET_True_useDTT_False_useSC_False/accs.pth'
acc_list = torch.load(Path_With_DET)
acc_list5 = [acc_list[i].item() for i in range(len(acc_list))]

Path_With_DET_DTT = root + 'snn_VthHand-1.0_useDET_True_useDTT_True_useSC_False/accs.pth'
acc_list = torch.load(Path_With_DET_DTT)
acc_list6 = [acc_list[i].item() for i in range(len(acc_list))]

plt.figure()
fig, ax = plt.subplots()
# ax.set_aspect(180)
ax.plot(acc_list1, 'y')
ax.plot(acc_list2, 'c')
ax.plot(acc_list3, 'b')
ax.plot(acc_list4, 'g')
ax.plot(acc_list5, 'r')
ax.plot(acc_list6, 'm')
ax.axhline(acc_target, color='k', linestyle='--')

ax1 = ax.inset_axes([0.55, 0.55, 0.3, 0.2])
ax1.plot(acc_list1, 'y')
ax1.plot(acc_list2, 'c')
ax1.plot(acc_list3, 'b')
ax1.plot(acc_list4, 'g')
ax1.plot(acc_list5, 'r')
ax1.plot(acc_list6, 'm')
ax1.plot(acc_list_target, color='k', linestyle='--')
ax1.set_xlim(224, 256)
if model_name == "vgg16":
    ax1.set_ylim(0.765, 0.788)
if model_name == "resnet20":
    ax1.set_ylim(0.8, 0.808)
ax.indicate_inset_zoom(ax1)

if model_name == "vgg16":
    plt.legend(['0.5 x vth', '0.7 x vth', '0.9 x vth', 'With DTT',
                'With DET', 'With DTT DET', 'Target Acc: {}'.format(acc_target)], fontsize=10, bbox_to_anchor=[0.62, 0.42, 0, 0])
if model_name == "resnet20":
    plt.legend(['0.5 x vth', '0.7 x vth', '0.9 x vth', 'With DTT',
                'With DET', 'With DTT DET', 'Target Acc: {}'.format(acc_target)], fontsize=10, bbox_to_anchor=[1.0, 0.42, 0, 0])
# plt.title('Spiking VGG16 on CIFAR100 Dataset')
plt.ylim([0, 0.81])
plt.xlim([0, 256])
plt.xlabel('Time Step', fontsize=11)
plt.ylabel('Top-1 Acc', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.show()
plt.savefig('{}/acc_cifar100_{}.pdf'.format(root, model_name), dpi=800)
print('done')

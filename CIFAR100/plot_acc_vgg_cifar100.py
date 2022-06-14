# plot for VGG-cifar100
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

acc_target = 0.7849
acc_list_target = [acc_target] * 256

Path_P_Norm = '/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_channelnormFalse_LIPoolingFalse_Burst1_SpicalibFalse256/accs.pth'
acc_list = torch.load(Path_P_Norm)
acc_list1 = [acc_list[i].item() for i in range(len(acc_list))]

Path_Burst = '/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_channelnormFalse_LIPoolingFalse_Burst5_SpicalibFalse256/accs.pth'
acc_list = torch.load(Path_Burst)
acc_list2 = [acc_list[i].item() for i in range(len(acc_list))]

Path_MLIPooling = '/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_channelnormFalse_LIPoolingTrue_Burst5_SpicalibFalse256/accs.pth'
acc_list = torch.load(Path_MLIPooling)
acc_list3 = [acc_list[i].item() for i in range(len(acc_list))]

Path_With_SpiCalib = '/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_channelnormFalse_LIPoolingFalse_Burst5_SpicalibTrue64/accs.pth'
acc_list = torch.load(Path_With_SpiCalib)
acc_list4 = [acc_list[i].item() for i in range(len(acc_list))]

Path_With_SpiCalib_MLIPooling = '/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_channelnormFalse_LIPoolingTrue_Burst5_SpicalibTrue128/accs.pth'
acc_list = torch.load(Path_With_SpiCalib_MLIPooling)
acc_list5 = [acc_list[i].item() for i in range(len(acc_list))]

plt.figure()
fig, ax = plt.subplots()
# ax.set_aspect(180)
ax.plot(acc_list1, 'y')
ax.plot(acc_list2, 'c')
ax.plot(acc_list3, 'b')
ax.plot(acc_list4, 'g')
ax.plot(acc_list5, 'r')
ax.axhline(acc_target, color='k', linestyle='--')

ax1 = ax.inset_axes([0.28, 0.5, 0.46, 0.32])
ax1.plot(acc_list1, 'y')
ax1.plot(acc_list2, 'c')
ax1.plot(acc_list3, 'b')
ax1.plot(acc_list4, 'g')
ax1.plot(acc_list5, 'r')
ax1.plot(acc_list_target, color='k', linestyle='--')
ax1.set_xlim(200, 256)
ax1.set_ylim(0.775, 0.788)
ax.indicate_inset_zoom(ax1)

plt.legend(['P-Norm', 'Burst Spikes', 'With MLIPooling', 'With SpiCalib',
            'With Spicalib and MLIPooLing', 'Target Acc: 0.7849'], fontsize=10, bbox_to_anchor=[0.47, 0.37, 0, 0])
# plt.title('Spiking VGG16 on CIFAR100 Dataset')
plt.ylim([0, 0.8])
plt.xlim([0, 256])
plt.xlabel('Time Step', fontsize=11)
plt.ylabel('Top-1 Acc', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.show()
plt.savefig('./result_conversion_vgg16/acc.pdf', dpi=800)
print('done')

import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

acc_list = torch.load('/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_LIPoolingTrue_Burst5_SpicalibTrue128/accs.pth')
acc_list1 = [acc_list[i].item() for i in range(len(acc_list))]

acc_list = torch.load('/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p1.0_LIPoolingTrue_Burst1_SpicalibFalse256/accs.pth')
acc_list2 = [acc_list[i].item() for i in range(len(acc_list))]

acc_list = torch.load('/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_LIPoolingTrue_Burst1_SpicalibFalse256/accs.pth')
acc_list3 = [acc_list[i].item() for i in range(len(acc_list))]

acc_list = torch.load('/data1/hexiang/newframework/conversion3/CIFAR100/result_conversion_vgg16/snn_timestep256_p0.995_LIPoolingTrue_Burst5_SpicalibFalse256/accs.pth')
acc_list4 = [acc_list[i].item() for i in range(len(acc_list))]

plt.figure()
fig, ax = plt.subplots()
# ax.set_aspect(180)
ax.plot(acc_list4, 'k')
ax.plot(acc_list2, 'b')
ax.plot(acc_list3, 'g')
ax.plot(acc_list1, 'r')

ax1 = ax.inset_axes([0.28, 0.36, 0.46, 0.32])
ax1.plot(acc_list4, 'k')
ax1.plot(acc_list2, 'b')
ax1.plot(acc_list3, 'g')
ax1.plot(acc_list1, 'r')
ax1.set_xlim(350, 450)
ax1.set_ylim(0.96, 0.98)
ax.indicate_inset_zoom(ax1)

plt.legend(['w/o Inhibition and STP', 'w/o Inhibition', 'w/o STP',  'w/ Inhibition and STP'], fontsize=10, bbox_to_anchor=[0.97, 0.27, 0, 0])
# plt.title('Spiking VGG16 on CIFAR100 Dataset')
plt.ylim([0, 0.8])
plt.xlim([0, 256])
plt.xlabel('Time Step', fontsize=11)
plt.ylabel('Top-1 Acc', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.show()
plt.savefig('./acc.pdf', dpi=600)
print('done')

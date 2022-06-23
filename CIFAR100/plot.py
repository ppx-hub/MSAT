import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from matplotlib.pyplot import MultipleLocator



# # 一元一次函数图像
# x = np.arange(-10, 10, 0.1)
# y = 0.3 * x + np.log(1 + np.exp(x / 1.0))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x, y)
# plt.show()
# plt.savefig("test_hx.jpg")
# sys.exit()


no_16 = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group3/snn_p1_VthHand1.0_useDET_False_useDTT_False"
path = os.path.join(root, "result_SINRate.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        no_16.append(list(map(float, numbers))[0])

have_16 = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group1/snn_p1_VthHand1.0_useDET_False_useDTT_False"
path = os.path.join(root, "result_SINRate.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        have_16.append(list(map(float, numbers))[0])

have_32 = []
root = "/home/hexiang/MSAT/CIFAR100/result_conversion_vgg16/parameters_group2/snn_p1_VthHand1.0_useDET_False_useDTT_False"
path = os.path.join(root, "result_SINRate.txt")
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        have_32.append(list(map(float, numbers))[0])
index = np.arange(1, 14)

fig, ax = plt.subplots()
bar_width = 0.3

ax.bar(index, no_16, bar_width, color='b', label='timestep 256')
ax.bar(index + bar_width, have_16, bar_width, color='m', label='timestep 16')
ax.bar(index + bar_width * 2, have_32, bar_width, color='r', label='timestep 32')
ax.legend()

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xlabel("layer index")
ax.set_ylabel("sin ratio")
ax.set_title('sin ratio in each VGG16 layer')

plt.show()
plt.savefig("./sin_ratio.pdf")
sys.exit()


Vth = []
Vmem = []
Vrd = []
plt.figure()
fig, ax = plt.subplots()

with open('Vmem_timestep.txt', 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data

    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        Vmem.append(list(map(float, numbers))[1])  # 转化为浮点数
        if ind == 255:
            ax.plot(Vmem, 'g')
            break

with open('Vrd_timestep.txt', 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data

    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        Vrd.append(list(map(float, numbers))[1])  # 转化为浮点数
        if ind == 255:
            ax.plot(Vrd, 'k')
            break

with open('Vth_timestep.txt', 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data

    for ind, line in enumerate(data):
        numbers = line.split()  # 将数据分隔
        if ind > 0:
            Vth.append(list(map(float, numbers))[1])  # 转化为浮点数
        if ind == 255:
            ax.plot(Vth, 'r')
            break


plt.legend(['Vmem', 'Vrd', 'Vth'], fontsize=10)
# plt.title('Spiking VGG16 on CIFAR100 Dataset')
plt.xlabel('Time Step', fontsize=11)
plt.ylabel('Value', fontsize=11)

plt.savefig('./vth.pdf', dpi=600)
print('done')

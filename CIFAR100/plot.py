import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 一元一次函数图像
x = np.arange(-10, 10, 0.1)
y = 0.8 * x + np.log(1 + np.exp(x / 1.0))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.show()
plt.savefig("test_hx.jpg")
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

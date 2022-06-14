from turtle import color
import matplotlib
from matplotlib import markers
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

'''
all-2-256：   4.082165    0.7867879746835443
all-3-256：   8.355107    0.786689082278481
-spicalib:   57.317192    0.7862935126582279
-LIPooling: 345.866302    0.7800632911392406
-gamma:    1085.394409    0.770371835443038
-p-norm:   4912.201172    0.7286392405063291
'''


# plt.scatter(0.7286392405063291, np.log10(4912.201172), s=10, marker='o', c='g')
plt.scatter(0.770371835443038, np.log10(1085.394409), linewidths=10, marker='o', c='g')
plt.scatter(0.7800632911392406, np.log10(345.866302), linewidths=10, marker='o', c='g')
plt.scatter(0.7862935126582279, np.log10(57.317192), linewidths=10, marker='o', c='g')
plt.scatter(0.786689082278481, np.log10(8.355107), linewidths=10, marker='o', c='g')
plt.scatter(0.7867879746835443, np.log10(4.082165), linewidths=10, marker='*', c='r')


plt.plot([0.7849, 0.7849], [0, 10], '--', color='b', linewidth=4)
plt.xlabel('Accuracy')
plt.ylabel('KL_Div')
plt.savefig('result/performance/p.pdf')



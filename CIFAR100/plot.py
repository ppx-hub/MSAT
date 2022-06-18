import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

f=open('Vth_timestep.txt', encoding='gbk')
txt=[]
for line in f:
    txt.append(line.strip())
print(txt)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x_values = [0, 1, 2, 3, 4, 5, 6, 7]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
x_np = torch.from_numpy(x_train)

y_values = [53807, 55217, 55209, 55415, 63100, 63206, 63761, 65766]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

slope, intercept, r, p, std_err = stats.linregress(x_values, y_values)

def func(x):
    return slope * x + intercept
model = list(map(func, x_values))

area = np.pi*3
colors = (0,0,0)

plt.scatter(x_values, y_values, s=area, c=colors, alpha=1)
plt.plot(x_values, model)
plt.show()

# test by Shawn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import matplotlib.pyplot as plt

x_values = [0, 1, 2, 3, 4, 5, 6, 7]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [53807, 55217, 55209, 55415, 63100, 63206, 63761, 65766]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
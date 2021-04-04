import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import matplotlib.pyplot as plt

x_values = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [53807, 55217, 55209, 55415, 63100, 63206, 63761, 65766]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
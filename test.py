from enum import Enum

import numpy as np
import torch


class enum(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

a = enum(0)
print(a.value == enum.EInfo.value)

a = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to('cpu')
print(a[3:5])
print(a[(torch.arange(a.shape[0]) < 3) + (torch.arange(a.shape[0]) >= 5)])

a1 = np.zeros(shape=(5, 7))
a2 = np.zeros(shape=(8, 7))
print(np.concatenate((a1, a2), axis=0).shape)
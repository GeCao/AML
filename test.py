from enum import Enum
import torch


class enum(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

a = enum(0)
print(a.value == enum.EInfo.value)

a = torch.Tensor([0, 1, 2, 3]).to('cpu')
b = torch.Tensor([0, 1, 2, 3]).to('cpu')
print(a*b)
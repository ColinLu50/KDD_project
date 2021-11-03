import numpy as np
import torch

row = torch.Tensor([1, 2]).long()
col = torch.Tensor([1, 2]).long()
index = torch.stack([row, col])
data = torch.FloatTensor([0.1, 0.3])
a = torch.sparse.FloatTensor(index, data, torch.Size((3, 3)))
print(a)

b = a.coalesce().cuda()
print(b)





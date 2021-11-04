import numpy as np
import torch

a = torch.zeros(1, 5)
a[0, 3] = 1

b = torch.zeros(1, 5)
b[0, 1] = 1

c = torch.cat([a, b], dim=0)
print(c)
print(c.shape)





import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DROLoss(nn.Module):
    
    def __init__(self, weight=None, epsilon=1):
        super(DROLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - 2*self.epsilon
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


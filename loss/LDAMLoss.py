import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_count=None):
        super(LDAMLoss, self).__init__()
        if cls_count is not None:
            print("initializing ldam loss with dependent C")
            m_list = 1.0 / np.sqrt(np.sqrt(cls_count))
            # original for class with 5 samples is 0.74
            m_list = m_list * (0.5 / np.max(m_list))
            m_list = torch.cuda.FloatTensor(m_list)
            self.m_list = m_list
            self.weight = 1/torch.FloatTensor(cls_count)
        else:
            print("initializing ldam loss with constant")
            self.m_list = torch.tensor([0.2]*1000).cuda()
            self.weight = None
        # self.m_list = torch.tensor([0.1]*1000).cuda()

    def forward(self, x, target):
        # create one hot encoding
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        # calculate margin based on batch
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        # keep  x for original ce
        return F.cross_entropy(output, target, weight=self.weight.cuda())
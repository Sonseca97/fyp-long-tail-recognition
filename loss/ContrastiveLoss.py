import torch 
import torch.nn as nn
import numpy as np 

class ContraLoss(torch.nn.Module):
    def __init__(self, p=2):
        super(ContraLoss, self).__init__()
        self.p = p

    def forward(self, inputs, centroids):
        loss = torch.mean(nn.PairwiseDistance(p=self.p)(inputs, centroids))
        return loss

if __name__ == '__main__':
    a = torch.tensor([[1.,1.],[2.,2.]])
    b = torch.tensor([[2., 2.], [4.,4.]])
    d = nn.PairwiseDistance(p=2)(a, b )
    print(d)
    print(d.shape)
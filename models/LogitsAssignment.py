import torch 
import torch.nn as nn 
import numpy as np 

class LogitsAssignment(nn.Module):

    def __init__(self):
        super(LogitsAssignment, self).__init__()
        
        self.layer1 = nn.Linear(2000, 1000)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1000, 1)

    def forward(self, ibs_logits, knn_logits, phase='train'):

        x = torch.cat([ibs_logits, knn_logits], axis=1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.squeeze(1)
        return x

def create_model():
    print("Loading Logits Assignment Module")
    lasm = LogitsAssignment()
    return lasm

if __name__ == '__main__':
    clf = LogitsAssignment()
    
    logits1 = torch.rand(128, 1000)
    logits2 = torch.rand(128, 1000)
    # print(clf(feature, logits).shape)
    clf(logits1, logits2)




import torch.nn as nn
import pandas as pd
import torch
def create_loss ():
    print("Loading MSE Loss")
    return nn.MSELoss()

if __name__ == "__main__":
    f = torch.tensor([
        [1,2,3],
        [2,3,4]
    ])

    m = torch.tensor([
        [1,1,1],
        [2,2,2],
        [3,3,3]
    ])
    print(f.shape)
    print(f.view(2, -1).shape)
    label = torch.tensor([2,3])
    label_expand = label.unsqueeze(1).expand(2, 10)
    
    classes = torch.arange(10).long()
    mask = label_expand.eq(classes.expand(2, 10))
    print(mask)
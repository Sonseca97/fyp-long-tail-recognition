import torch 
import torch.nn as nn 

class LogitsWeight(nn.Module):

    def __init__(self, num_classes=1000):
        super(LogitsWeight, self).__init__()
        
        self.register_parameter(name='logitsweight', param=torch.nn.Parameter(torch.randn(num_classes)))
        self.register_parameter(name='logitsbias', param=torch.nn.Parameter(torch.randn(num_classes)))


    def forward(self, logits):
        return self.logitsweight * logits + self.logitsbias

def create_model(num_classes=1000):
    print("Loading Logits Weight")
    w = LogitsWeight(num_classes=num_classes)
    return w

if __name__ == '__main__':
    clf = LogitsWeight()
    logits = torch.rand(128, 1000)
    print(clf.parameters())
import torch 
import torch.nn as nn 

class DistributionAlignment(nn.Module):

    def __init__(self, feat_dim=512, num_classes=1000):
        super(DistributionAlignment, self).__init__()
        
        self.register_parameter(name='logitsweight', param=torch.nn.Parameter(torch.randn(num_classes)))
        self.register_parameter(name='logitsbias', param=torch.nn.Parameter(torch.randn(num_classes)))
        self.linear = nn.Linear(feat_dim, 1)


    def forward(self, features, logits):

        w1 = torch.sigmoid(self.linear(features))
        return w1 * (self.logitsweight * logits + self.logitsbias) + (1 - w1) * logits

def create_model(feat_dim=512, num_classes=1000):
    print("Loading Logits Weight")
    w = DistributionAlignment(feat_dim=feat_dim, num_classes=num_classes)
    return w

if __name__ == '__main__':
    clf = LogitsWeight()
    logits = torch.rand(128, 1000)
    print(clf.parameters())
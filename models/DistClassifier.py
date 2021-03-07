import torch 
import torch.nn as nn 
# from utils import * 

class Dist_Classifier(nn.Module):

    def __init__(self, feat_dim=2048, num_classes=1000):
        super(Dist_Classifier, self).__init__()
        self.num_classes = num_classes 
        self.feat_dim = feat_dim
        self.fc = nn.Linear(feat_dim, feat_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc(x)


def create_model(feat_dim=2048, num_classes=1000, test=False):
    print("Loading Distance Classifier.")
    clf = Dist_Classifier(feat_dim, num_classes)

    if not test:
        print("Random initialized classifier weights.")
    

    return clf 
    
if __name__ == "__main__":
    clf = create_model(feat_dim=512)
    a = torch.rand((128, 512))
    print(clf(a).shape)


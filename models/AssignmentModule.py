import torch 
import torch.nn as nn 
import numpy as np 

class AssignmentModule(nn.Module):

    def __init__(self, feat_dim=512, top_k=30, layer_feat_norm_dim=16):
        super(AssignmentModule, self).__init__()
        
        self.layer_feat_norm = nn.Linear(feat_dim, layer_feat_norm_dim)

        self.pred_fc = nn.Linear(layer_feat_norm_dim + top_k, 1)
        # self.pred_fc = nn.Linear(layer_feat_norm_dim, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.top_k = top_k

    def mixup(self, x, alpha=1):
        '''Returns mixed inputs'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x

    def forward(self, normalized_features, logits, knn_logits, labels, phase='train'):

        correctness_linear = logits.argmax(dim=1) == labels
        correctness_knn = knn_logits.argmax(dim=1) == labels
        target = (~correctness_linear)&correctness_knn
        
        # # positive indexes 
        # pos_idx = torch.nonzero(target==True).view((-1,))


        # make balance
        # if len(pos_idx)!=0 and phase=='train' and 1 == 2:
        #     ratio = int((normalized_features.shape[0] - len(pos_idx)) / len(pos_idx))
        #     augmented_features = torch.cat(ratio * [normalized_features[pos_idx]])
        #     augmented_logits = torch.cat(ratio * [logits[pos_idx]])
        #     augmented_targets = torch.cat(ratio * [target[pos_idx]])
            
        #     final_input = torch.cat([normalized_features, augmented_features])
        #     final_logits = torch.cat([logits, augmented_logits])
        #     final_target = torch.cat([target, augmented_targets])

        #     feature = self.relu(self.layer_feat_norm(final_input)) # 128 * 512
        #     topk, _ = torch.topk(final_logits, k=self.top_k, dim=1)

        #     concat_input = torch.cat((topk, feature), dim=1)
        #     output = self.pred_fc(concat_input).view((-1,))

        #     return output, final_target

        # else:
        feature = self.leaky_relu(self.layer_feat_norm(normalized_features)) # 128 * 512
        topk, _ = torch.topk(logits, k=self.top_k, dim=1)

        concat_input = torch.cat((topk, feature), dim=1)
        output = self.pred_fc(concat_input).view((-1,))

        return output, target

def create_model(feat_dim=512, topk=100, layer_feat_norm_dim=50):
    print("Loading Assignment Module")
    asm = AssignmentModule(feat_dim, topk, layer_feat_norm_dim)
    return asm

if __name__ == '__main__':
    clf = AssignmentModule()
    feature = torch.rand(128, 512)
    logits = torch.rand(128, 1000)
    # print(clf(feature, logits).shape)
    correctness = torch.tensor([False, True, True, False])
    correct_knn = torch.tensor([True, False, True, False])
    target = (~correctness) & correct_knn
    print(target)
    def func(a):
        print("haha")
        return 
        print("nice")
    func(target)





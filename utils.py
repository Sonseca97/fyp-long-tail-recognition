import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, classification_report
import importlib
import pdb
import math
import pickle as pkl 
import torch.nn.functional as F

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model

def shot_acc (preds, preds_topk, labels, train_data, many_shot_thr=100, low_shot_thr=20):

    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = [] # 1000
    class_correct = [] # 1000
    class_correct_topk = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l])) # 1000
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
        class_correct_topk.append(mic_acc_cal_topk(preds_topk[labels == l].cpu(),torch.from_numpy(labels[labels == l])))

    many_shot = []
    median_shot = []
    low_shot = []
    many_shot_topk = []
    median_shot_topk = []
    low_shot_topk = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
            many_shot_topk.append(class_correct_topk[i])
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
            low_shot_topk.append(class_correct_topk[i])
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
            median_shot_topk.append(class_correct_topk[i])
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_correct, test_class_count, \
        np.mean(many_shot_topk), np.mean(median_shot_topk), np.mean(low_shot_topk)

def shot_acc_expr (preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):

    training_labels = np.array(train_data.dataset.labels).astype(int)

    labels = labels.detach().cpu().numpy()
    train_class_count = []

    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l])) # 1000

    low_shot_class = []
    for i in range(len(train_class_count)):
        if train_class_count[i] <= low_shot_thr:
            low_shot_class.append(i)
    return low_shot_class
    
def get_cls_report(labels, preds):
    return classification_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

def F_measure(preds, labels, openset=False, theta=None):

    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.

        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1

def mic_acc_cal_topk(preds, labels, topk=10):
    correct = preds.eq(labels.view(-1, 1).expand_as(preds))
    result = []
    # for k in range(1, topk+1):
    #     correct_k = correct[:, :k]
    #     result.append(correct_k.sum(axis=1).sum().item()/len(labels))
    # print(result)
    # exit()
    return correct.sum(axis=1).sum().item()/len(labels)

def class_count (data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

def l2_similarity(A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.shape[0]

        AB = torch.mm(A, B.t())
        AA = (A**2).sum(dim=1, keepdim=True)
        BB = (B**2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2*AB

        return -dist

def cos_similarity(A, B):
        feat_dim = A.size(1)
        AB = torch.mm(A, B.t())
        AB = AB / feat_dim
        return AB

def scaling(A, B):
    dim = A.shape[0]
    #(N_max - N_min)/(O_max - O_min) * (O - O_min) + N_min
    N_max, _ = B.max(axis=1)
    N_min, _ = B.min(axis=1)
    O_max, _ = A.max(axis=1)
    O_min, _ = A.min(axis=1)

    #(N_max - N_min)/(O_max - O_min)
    division = ((N_max - N_min)/(O_max - O_min)).reshape((dim,1))
    # * (O - O_min)
    mul = A - O_min.reshape((dim, 1))
    # + N_min
    return division * mul + N_min.reshape((dim,1))

def matrix_norm(A):
    return F.normalize(A, p=2, dim=1)


def compute_center_loss(features, centers, targets, loss, device):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    assert loss in ['cos', 'mse'], 'Please specify either cosine loss or mse loss'
    if loss == 'cos':
        criterion = torch.nn.CosineEmbeddingLoss()
        center_loss = criterion(features, target_centers, torch.ones(features.shape[0]).to(device))
    else:
        criterion = torch.nn.MSELoss()
        center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha, device, head, median, tail):
  
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)
    
  

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    '''
        create mask
        HEAD: 0.999
        MEDIAN/TAIL: 0.99
    '''
    # alpha = torch.tensor([0.01]*uni_targets.shape[0]).to(device)
    # mask_head = (uni_targets.unsqueeze(1) == torch.tensor(tail).to(device)).any(-1)
    # alpha[mask_head] = 0.2
    # alpha = alpha.unsqueeze(1)
   
    # delta_centers = delta_centers / (same_class_feature_count+1) * alpha

    delta_centers = (delta_centers / (same_class_feature_count)) * alpha
  
    result = torch.zeros_like(centers)

   
    result[uni_targets, :] = delta_centers

    return result
    
# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""

#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))

#     return distribution


def get_shot_list(path='ImageNet_shots.pkl'):
    with open(path, 'rb') as f:
        shot_list = pkl.load(f)
        # shot_list['many_median_shot'] = shot_list['many_shot'] + shot_list['median_shot']
        return shot_list

def map_classid_and_label(shot_list):
    classid2label = {}
    label2classid = {}
    for label, classid in enumerate(shot_list):
        classid2label[classid] = label
        label2classid[label] = classid

    return classid2label, label2classid

def cal_attention(q, k, v):
    '''
    q: batch features
    k: memory module
    v: batch logits (batch_size, C)
    '''
    w = q.mm(k.t())
    w = F.softmax(w, dim=1)

    return w * v

def cal_attention_topk(q, k, v):
    '''
    q: batch features
    k: memory module
    v: batch logits (batch_size, C)
    '''
    w = q.mm(k.t())
    topk_v. topk_idx = torch.topk(v, dim=1, k=5)
    w = torch.gather(w, dim=1, index=topk_idx)



    return w * topk_v, topk_idx



if __name__ == '__main__':
    features = torch.tensor([[1.,2.,3.], [2.,1.,3.], [3.,3.,3]])
    features_exp = torch.tensor([[2.,2.5,3.], [2.,1.,3.], [0.,0.,0.]])
    centers = torch.tensor([[9.,5.,3.], [3.,2.,1.], [3.,3.,4.]])
    targets = torch.tensor([0,1,0])
    alpha = 0.01
    device = torch.device('cpu')
    center_delta = get_center_delta(features, centers, targets, alpha, device, None, None, None)
    new_centers = centers - center_delta
    print(center_delta)
    print(new_centers)
    print(0.99 * centers + 0.01 * features_exp)


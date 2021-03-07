import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, classification_report
import importlib
import pdb
import math
import torch.nn.functional as F


def mixup_rank(x, y, label_dict, count, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # convert label to list
    # create sorted list by label distribution
    label_list = y.tolist()
    list_high_low = sorted(label_list, key=label_dict.get, reverse=True)
    list_low_high = sorted(label_list, key=label_dict.get)

    # label order from high to low
    y_high_low = torch.tensor(list_high_low).cuda()
    index_high_low = y.argsort()[y_high_low.argsort().argsort()]

    #label order from low to high
    y_low_high = torch.tensor(list_low_high).cuda()
    index_low_high = y.argsort()[y_low_high.argsort().argsort()]

    # sort x using index
    x_high_low = x[index_high_low, :]
    x_low_high = x[index_low_high, :]
    
    # mixup
    mixed_x = lam * x_high_low + (1 - lam) * x_low_high
    y_a, y_b = y_high_low, y_low_high
    return mixed_x, y_a, y_b, lam, count, y_a

# Mixup of Highest frequent to Lowest frequent
def mixup_rank_enrich(x, y, label_dict, count, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # convert label to list
    # create sorted list by label distribution
    label_list = y.tolist()
    list_high_low = sorted(label_list, key=label_dict.get, reverse=True)
    list_low_high = sorted(label_list, key=label_dict.get)

    # label order from high to low
    y_high_low = torch.tensor(list_high_low).cuda()
    index_high_low = y.argsort()[y_high_low.argsort().argsort()]

    #label order from low to high
    y_low_high = torch.tensor(list_low_high).cuda()
    index_low_high = y.argsort()[y_low_high.argsort().argsort()]

    # sort x using index
    x_high_low = x[index_high_low, :]
    x_low_high = x[index_low_high, :]
    
    # mixup
    mixed_x = lam * x_high_low + (1 - lam) * x_low_high
    y_a, y_b = y_high_low, y_low_high
    return mixed_x, y_a, y_b, lam, count, y_a
    # concat with original to create enriched batch
    final_x = torch.cat((x, mixed_x), dim=0)
    final_y_a = torch.cat((y, y_a), dim=0)
    final_y_b = torch.cat((y, y_b), dim=0)

    return final_x, final_y_a, final_y_b, lam, count, final_y_a


# mixup of tail with other class enrich
def mixup_tail_enrich(x, y, label_dict, count, alpha=0.3):
    tail = [k for k in label_dict.keys() if label_dict[k]<=20]
    f=32
    # return torch.cat([x,x]), torch.cat([y,y]), torch.cat([y,y]), -1, count, torch.cat([y,y])
    #split to head-tail
    tail_ts = []
    rest_ts = [] # non-taill images
    rest_lb = []
    tail_lb = []
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)
    
    # if tail class is sampled from this minibatch
    if len(tail_ts) != 0:
        for i in tail_lb:
            count[i] += 1
        tail_ts = torch.stack(tail_ts, dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)
        rest_ts = torch.stack(rest_ts, dim=0)
        rest_lb = torch.stack(rest_lb, dim=0)
        for i in rest_lb:
            count[i] += 1
        mix_index = torch.randperm(len(tail_lb)).cuda()
        index = torch.randperm(len(rest_lb)).cuda()[:len(tail_lb)]
        
        mixed_x = lam * tail_ts[mix_index,:] + (1 - lam) * rest_ts[index,:]
        mixed_y_a, mixed_y_b = tail_lb[mix_index], rest_lb[index]
        
        if len(tail_ts)<32:
            factor = math.ceil(f / len(tail_ts))
            for n in range(1, factor):
                mix_index = torch.randperm(len(tail_ts)).cuda()
                index = torch.randperm(len(rest_lb)).cuda()[:len(tail_ts)]
                tmp_x = lam * tail_ts[mix_index,:] + (1 - lam) * rest_ts[index,:]
                tmp_y_a, tmp_y_b = tail_lb[mix_index], rest_lb[index]
                mixed_x = torch.cat((mixed_x, tmp_x),dim=0)
                mixed_y_a, mixed_y_b = torch.cat((mixed_y_a, tmp_y_a),dim=0), torch.cat((mixed_y_b, tmp_y_b),dim=0)
        
        mixed_x = mixed_x[:f,:]
        mixed_y_a, mixed_y_b = mixed_y_a[:f], mixed_y_b[:f]
        for i in mixed_y_a:
            count[i] += lam 
        for i in mixed_y_b:
            count[i] += 1 - lam
        x = torch.cat((mixed_x, tail_ts, rest_ts),dim=0)
        y_a = torch.cat((mixed_y_a, tail_lb, rest_lb),dim=0)
        y_b = torch.cat((mixed_y_b, tail_lb, rest_lb), dim=0)
        return x, y_a, y_b, lam, count, y_a
    
    else:
        lam = -1
        return x, y, y, lam, count, y


# mixup of head class with other classes
def mixup_head_enrich(x, y, tail, count, alpha=0.3):
    f = 32
    #split to head-tail
    tail_ts = []
    rest_ts = [] # non-taill images
    rest_lb = []
    tail_lb = []
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)

    # if tail class is sampled from this minibatch
    if len(tail_ts) != 0:
        for i in tail_lb:
            count[i] += 1
        tail_ts = torch.stack(tail_ts, dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)
    else:
        return x, y, y, lam, count, y

    # if head class is sampled from this minibatch
    # perform mixup on head class
    if len(rest_ts)!=0:
        rest_ts = torch.stack(rest_ts,dim=0)
        rest_lb = torch.stack(rest_lb, dim=0)
        for i in rest_lb:
            count[i] += 1
        batch_size = x.size()[0]
        mix_ts = []
        mix_lb = []
        mix_index = torch.randperm(f).cuda()
        index = torch.randperm(batch_size).cuda()[:f]

        for i in y[index]:
            count[i] += 1-lam
        for i in rest_lb[mix_index]:
            count[i] += lam

        mixed_rest_x = lam * rest_ts[mix_index,:] + (1 - lam) * x[index, :]

        mixed_rest_y_a, mixed_rest_y_b = rest_lb[mix_index], y[index]

        x = torch.cat((mixed_rest_x, tail_ts, rest_ts),dim=0)
        y_a = torch.cat((mixed_rest_y_a, tail_lb, rest_lb),dim=0)
        y_b = torch.cat((mixed_rest_y_b, tail_lb, rest_lb), dim=0)

        return x, y_a, y_b, lam, count, y_a

    else:
        return tail_ts, tail_lb, tail_lb, lam, count, y

# mixup of head strictly with tail
def mixup_head_with_tail(x, y, tail, count, alpha=1):
    tail_ts = []
    rest_ts = [] # non-taill images
    rest_lb = []
    tail_lb = []
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)
    rest_ts = torch.stack(rest_ts, dim=0)
    rest_lb = torch.stack(rest_lb, dim=0)
    batch_size = x.size()[0]
    rest_size = rest_ts.size()[0]

    # if tail class is sampled from this minibatch
    if len(tail_ts) != 0:
        for i in tail_lb:
            count[i] += 1
        tail_ts = torch.stack(tail_ts, dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)
    else:
        return x, y, y, lam, count

    # if head class and tail class both sampled
    # mixup tail_size of head with tail
    # keep the remaining tail
    tail_size = len(tail_ts)
    index_tail = torch.randperm(tail_size).cuda()
    index_head = torch.randperm(rest_size).cuda()
    selected_head = rest_ts[index_head,:]
    selected_tail = tail_ts[index_tail,:]
    mix_x = lam * selected_head + mix_
    for i in rest_lb:
        count[i] += lam
    mix_ts = []
    mix_lb = []
    index = torch.randperm(batch_size).cuda()[:tail_size]
    for i in y[index]:
        count[i] += 1-lam
    mixed_rest_x = lam * rest_ts + (1 - lam) * x[index, :]
    mixed_rest_y_a, mixed_rest_y_b = rest_lb, y[index]
    # if it has tail class, concat head and tail tgt
    if len(tail_ts) != 0:
        x = torch.cat((mixed_rest_x, tail_ts), dim=0)
        y_a = torch.cat((mixed_rest_y_a, tail_lb),dim=0)
        y_b = torch.cat((mixed_rest_y_b, tail_lb), dim=0)
        return x, y_a, y_b, lam, count
    # if not, just return head
    else:
        return mixed_rest_x, mixed_rest_y_a, mixed_rest_y_b, lam, count



# mixup of non-tail class with all other classes (head/median/tail)
def mixup_head(x, y, label_dict, count, alpha=1):
    tail = [k for k in label_dict.keys() if label_dict[k]<=20]
    tail_ts = []
    rest_ts = [] # non-taill images
    rest_lb = []
    tail_lb = []
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)

    # if tail class is sampled from this minibatch
    if len(tail_ts) != 0:
        for i in tail_lb:
            count[i] += 1
        tail_ts = torch.stack(tail_ts, dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)

    # if head class is sampled from this minibatch
    # perform mixup on head class
    if len(rest_ts)!=0:
        rest_ts = torch.stack(rest_ts,dim=0)
        rest_lb = torch.stack(rest_lb, dim=0)
        for i in rest_lb:
            count[i] += lam
        batch_size = x.size()[0]
        rest_size = rest_ts.size()[0]
        mix_ts = []
        mix_lb = []
        index = torch.randperm(batch_size).cuda()[:rest_size]
        for i in y[index]:
            count[i] += 1-lam
        mixed_rest_x = lam * rest_ts + (1 - lam) * x[index, :]
        mixed_rest_y_a, mixed_rest_y_b = rest_lb, y[index]
        # if it has tail class, concat head and tail tgt
        if len(tail_ts) != 0:
            x = torch.cat((mixed_rest_x, tail_ts), dim=0)
            y_a = torch.cat((mixed_rest_y_a, tail_lb),dim=0)
            y_b = torch.cat((mixed_rest_y_b, tail_lb), dim=0)
            return x, y_a, y_b, lam, count, torch.cat((rest_lb, tail_lb),dim=0)
        # if not, just return head
        else:
            return mixed_rest_x, mixed_rest_y_a, mixed_rest_y_b, lam, count, y

    else:
        return tail_ts, tail_lb, tail_lb, lam, count, y

# mixup tail class with all other images within the batch
def mixup_tail(x, y, label_dict, count, alpha=1):
    tail = [k for k in label_dict.keys() if label_dict[k]<=20]
    tail_ts = []
    rest_ts = [] # non-taill images
    rest_lb = []
    tail_lb = []
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)
    rest_ts = torch.stack(rest_ts, dim=0)
    rest_lb = torch.stack(rest_lb, dim=0)
    for i in rest_lb:
        count[i] += 1
    if len(tail_ts)!=0:
        tail_ts = torch.stack(tail_ts,dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)
        batch_size = x.size()[0]
        tail_size = tail_ts.size()[0]
        mix_ts = []
        mix_lb = []
        index = torch.randperm(batch_size).cuda()[:tail_size]
        mixed_tail_x = lam * tail_ts + (1 - lam) * x[index, :]
        for i in tail_lb:
            count[i] += lam
        for i in y[index]:
            count[i] += 1 - lam
        x = torch.cat((mixed_tail_x, rest_ts), dim=0)
        mixed_tail_y_a, mixed_tail_y_b = tail_lb, y[index]
        y_a = torch.cat((mixed_tail_y_a, rest_lb),dim=0)
        y_b = torch.cat((mixed_tail_y_b, rest_lb), dim=0)
        return x, y_a, y_b, lam, count, torch.cat((tail_lb, rest_lb), dim=0)
    else:
        return rest_ts, rest_lb, rest_lb, lam, count, rest_lb

#mixup tail with tail
def mixup_data(x, y, tail, count, alpha=0.8):
    rest_ts = [] # non-taill images
    rest_lb = [] # non-tail labels
    tail_ts = [] # tail images
    tail_lb = [] # tail labels
    for idx in range(len(y)):
        if y[idx].item() in tail:
            tail_lb.append(y[idx])
            tail_ts.append(x[idx])
        else:
            rest_ts.append(x[idx])
            rest_lb.append(y[idx])
    tail_ts, tail_lb, rest_ts, rest_lb = tuple(tail_ts),tuple(tail_lb),tuple(rest_ts), tuple(rest_lb)
    rest_ts = torch.stack(rest_ts, dim=0)
    rest_lb = torch.stack(rest_lb, dim=0)
    for i in rest_lb:
        count[i] += 1
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if len(tail_ts)!=0:
        tail_ts = torch.stack(tail_ts,dim=0)
        tail_lb = torch.stack(tail_lb, dim=0)
        '''Returns mixed inputs, pairs of targets, and lambda'''

        tail_size = tail_ts.size()[0]
        tail_index = torch.randperm(tail_size).cuda()

        mixed_tail_x = lam * tail_ts + (1 - lam) * tail_ts[tail_index, :]
        x = torch.cat((mixed_tail_x, rest_ts), dim=0)
        mixed_tail_y_a, mixed_tail_y_b = tail_lb, tail_lb[tail_index]
        for i in tail_lb:
            count[i] += lam
        for i in tail_lb[tail_index]:
            count[i] += 1- lam
        y_a = torch.cat((mixed_tail_y_a, rest_lb),dim=0)
        y_b = torch.cat((mixed_tail_y_b, rest_lb), dim=0)
        return x, y_a, y_b, lam, count, y
    else:
        return rest_ts, rest_lb, rest_lb, lam, count, y


def mixup(x, y, tail, count, alpha=0.3):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    for i in y_a:
        count[i] += lam
    for i in y_b:
        count[i] += 1 - lam
    return mixed_x, y_a, y_b, lam, count, y

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

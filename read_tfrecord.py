import torch
import torchvision
import torch.nn.functional as F
from torchvision import models
from tfrecord.torch.dataset import TFRecordDataset
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import cv2
import os
import numpy as np
import argparse
from torch import optim
from grid import *
MIXUP = False
GRIDMASK = False
NUMLABEL = 100
EPOCHS = 200

# SET GPU ID 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']='0'

#CONFIGURE DATA
tfrecord_train = "/home/lizhaochen/FYP/data/cifar-100-data-im-0.05/train.tfrecords"
tfrecord_val = "/home/lizhaochen/FYP/data/cifar-100-data-im-0.05/eval.tfrecords"
index_path = None
description = {"image": "byte", "label": "int"}
train_dataset = TFRecordDataset(tfrecord_train, index_path, description)
val_dataset = TFRecordDataset(tfrecord_val, index_path, description)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)


# function for MIX UP
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# INITIALIZE GRIDMASK
grid = GridMask(24, 32, 360, 0.4, 1, 0.8)

#
model = models.resnet18()
fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512,NUMLABEL)),
            # ('relu', nn.ReLU()),
            # ('fc2', nn.Linear(100,10)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
model.fc = fc
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


classes = list(range(NUMLABEL))
for epoch in range(EPOCHS):
    train_loss = 0
    validation_loss = 0
    train_correct = 0
    val_correct = 0
    num_train = 0
    num_test = 0
    totaltrain = 0
    model.train()
    if GRIDMASK:
        grid.set_prob(epoch, EPOCHS)
    class_correct = list(0. for i in range(NUMLABEL))
    class_total = list(0. for i in range(NUMLABEL))
    for i, data in enumerate(trainloader):
        num_train += 1
        totaltrain += data['label'].size(0)
        input, label = data['image'].to(device), data['label'].long().squeeze(1).to(device)
        if MIXUP:
            input, targets_a, targets_b, lam = mixup_data(input, label, 1.0)
            input, targets_a, targets_b = map(Variable, (input, targets_a, targets_b))
            outputs = model.forward(input)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            _, preds = torch.max(outputs,1)
            train_correct += lam * torch.sum(preds == targets_a.data).float() + (1 - lam) * torch.sum(preds == targets_b.data).float()
        else:
            if GRIDMASK:
                input = grid(input)
            outputs = model.forward(input)
            loss = criterion(outputs, label)
            _, preds = torch.max(outputs,1)
            # correct = (preds == label.data)
            train_correct += torch.sum(preds == label.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # for j in range(data['label'].size(0)):
        #     l = label.data[j]
        #     class_correct[l] += correct[j].item()
        #     class_total[l] += 1
    # for i in range(NUMLABEL):
    #     print('TRAINING progress (%2d/%2d)' % (
    #         np.sum(class_correct[i]), np.sum(class_total[i])))
    model.eval()
    with torch.no_grad():
        class_correct = list(0. for i in range(NUMLABEL))
        class_total = list(0. for i in range(NUMLABEL))
        totaltest = 0
        for data in valloader:
            totaltest += data['label'].size(0) #10000 for cifar-100
            num_test += 1
            input, label = data['image'].to(device), data['label'].long().to(device).squeeze(1)
            outputs = model.forward(input)
            val_loss = criterion(outputs, label)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(outputs, 1)
            val_correct += torch.sum(val_preds == label.data)
            correct = (val_preds == label.data)
    #         for j in range(data['label'].size(0)):
    #             l = label.data[j]
    #             class_correct[l] += correct[j].item()
    #             class_total[l] += 1
    # if epoch == 0 or epoch == 199:
    #     for i in range(NUMLABEL):
    #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #             classes[i], 100 * class_correct[i] / class_total[i],
    #             np.sum(class_correct[i]), np.sum(class_total[i])))
    train_loss = train_loss/ num_train
    train_acc = train_correct.double() / totaltrain
    validation_loss =  validation_loss / num_test
    val_acc = val_correct.double() / totaltest
    print("Epoch {}/{} ..Train loss:{}..Train accuracy:{}".format(str(epoch+1), str(200), '%.4f'%(train_loss), '%.4f'%(train_acc*100)))
    print("Epoch {}/{}.. Test loss: {}..Test accuracy: {}".format(str(epoch+1), str(200), '%.4f'%(validation_loss), '%.4f'%(val_acc*100)))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Configuration of setup and training process")
#     parser.add_argument('--train_path', type=str, help='path to training folder')
#     parser.add_argument('--val_path', type=str, help='path to validation folder')
#     parser.add_argument('--checkpoints_path', type=str, help='path to save models')
#     parser.add_argument('--epochs', type=int, default=500,help= 'number of epochs')
#     parser.add_argument('--lr', type= float, default=0.01, help= 'learning rate for training')
#     parser.add_argument('--batchsize', type=int, default=256, help= 'batchsize for training/validation')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum value')
#     args = parser.parse_args()

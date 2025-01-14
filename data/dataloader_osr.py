from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import re
from utils import *
# Data transformation with augmentation
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ## need keep??
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        if 'val' in path.split('/'):
            path = re.sub(r'n\d{8}\/', '', path)
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, path

class Shot_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, shot_phase=None):
        self.shot_list = get_shot_list()['many_shot'] + get_shot_list()['median_shot']
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.classid2label, self.label2classid = map_classid_and_label(self.shot_list)
        with open(txt) as f:
            for line in f:
                classid = int(line.split()[1])
                self.img_path.append(os.path.join(root, line.split()[0]))
                if classid in self.shot_list:
                    self.labels.append(self.classid2label[classid])
                else:
                    self.labels.append(len(self.shot_list))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        if 'val' in path.split('/'):
            path = re.sub(r'n\d{8}\/', '', path)
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path

# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=8, test_open=False, shuffle=True):

    txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))
 
    print('Loading data from %s' % (txt))
    if dataset != 'iNaturalist18':
        key = 'default'
    else:
        key = 'iNaturalist18'

    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    # print('Use data transformation:', transform)

    set_ = Shot_Dataset(data_root, txt, transform)

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size,
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)


class ASM_Dataset(Dataset):

    def __init__(self, normalized_features, logits, knn_logits, labels):
        self.normalized_features = normalized_features.cpu()
      
   
        self.labels = labels.cpu()
        self.logits = logits.cpu()
        self.knn_logits = knn_logits.cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
    
        normalized_features = self.normalized_features[index]
        labels = self.labels[index]
        logits = self.logits[index]
        knn_logits = self.knn_logits[index]
       
        return normalized_features, logits, knn_logits, labels
